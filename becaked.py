from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Activation, Concatenate, Add, Subtract, Multiply, Lambda, Reshape, Flatten, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.layers import Layer
import os
from data_utils import *
from utils import *
from generator import *

DAYS = 10
NUMBER_OF_HYPER_PARAM = 3

class SelfAttention(Layer):
    def __init__(self,
                 aspect_size,
                 hidden_dim,
                 penalty=1.0,
                 return_attention=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.aspect_size = aspect_size
        self.hidden_dim = hidden_dim
        self.penalty = penalty
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.return_attention = return_attention
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, Sequence_size, Sequence_hidden_dim)
        assert len(input_shape) >= 3
        batch_size, sequence_size, sequence_hidden_dim = input_shape

        self.Ws1 = self.add_weight(shape=(self.hidden_dim, sequence_hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.Ws2 = self.add_weight(shape=(self.aspect_size, self.hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.cast(K.shape(inputs)[0], K.floatx())
        inputs_t = K.permute_dimensions(inputs, (1,2,0)) # H.T
        d1 = K.tanh(K.permute_dimensions(K.dot(self.Ws1, inputs_t), (2,0,1))) # d1 = tanh(dot(Ws1, H.T))
        d1 = K.permute_dimensions(d1, (2,1,0))
        A = K.softmax(K.permute_dimensions(K.dot(self.Ws2, d1), (2,0,1))) # A = softmax(dot(Ws2, d1))
        H = K.permute_dimensions(inputs, (0,2,1))
        outputs = K.batch_dot(A, H, axes=2) # M = AH

        A_t = K.permute_dimensions(A, (0,2,1))
        I = K.eye(self.aspect_size)
        P = K.square(self._frobenius_norm(K.batch_dot(A, A_t) - I)) # P = (frobenius_norm(dot(A, A.T) - I))**2
        self.add_loss(self.penalty*(P/batch_size))

        if self.return_attention:
            return [outputs, A]
        else:
            return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 3
        assert input_shape[-1]
        batch_size, sequence_size, sequence_hidden_dim = input_shape
        output_shape = tuple([batch_size, self.aspect_size, sequence_hidden_dim])

        if self.return_attention:
            attention_shape = tuple([batch_size, self.aspect_size, sequence_size])
            return [output_shape, attention_shape]
        else: return output_shape


    def get_config(self):
        config = {
            'aspect_size': self.aspect_size,
            'hidden_dim': self.hidden_dim,
            'penalty':self.penalty,
            'return_attention': self.return_attention,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _frobenius_norm(self, inputs):
        outputs = K.sqrt(K.sum(K.square(inputs)))
        return outputs

def SIRD_layer(tensors):
    input_raw, x = tensors

    S = tf.subtract(
                input_raw[:,:,0],
                tf.multiply(
                    tf.multiply(
                        x[:,0],
                        input_raw[:,:,0]
                        ),
                    input_raw[:,:,1]
                    )
                )

    I = tf.subtract(
            tf.add(
                input_raw[:,:,1],
                tf.multiply(tf.multiply(x[:,0], input_raw[:,:,0]), input_raw[:,:,1])
                ),
            tf.multiply(
                tf.add(x[:,1], x[:,2]),
                input_raw[:,:,1]
                )
            )

    R = tf.add(
            input_raw[:,:,2],
            tf.multiply(
                x[:,1],
                input_raw[:,:,1]
                )
            )

    D = tf.add(
            input_raw[:,:,3],
            tf.multiply(
                x[:,2],
                input_raw[:,:,1]
                )
            )

    out = tf.stack([S, I, R, D], axis=-1)
    return out

def case_diff(tensor):
    return tf.subtract(tensor[:,1:], tensor[:,:-1])

class BeCakedModel():
    def __init__(self, population=7.5e9, day_lag=DAYS):
        self.initN = population
        self.day_lag = day_lag
        self.model = self.build_model(day_lag)

        if os.path.exists("models/world_%d.h5"%day_lag):
            self.load_weights("models/world_%d.h5"%day_lag)

        self.model.summary()
        self.estimator_model = Model(inputs=self.model.input,
                                        outputs=self.model.layers[-2].output)

    def update_population(self, population):
        self.initN = population

    def reset_population(self):
        self.initN = 7.5e9

    def load_weights(self, path):
        print("Loading saved model at %s"%path)
        self.model.load_weights(path)
        self.estimator_model = Model(inputs=self.model.input,
                                        outputs=self.model.layers[-2].output)

    def build_model(self, day_lag):
        inputs = Input(shape=(day_lag, 4)) # S, I, R, D

        acce = Lambda(case_diff)(inputs)
        enc_out, enc_state = tf.keras.layers.GRU(256,
                          # Return the sequence and state
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')(acce)

        att_out = SelfAttention(NUMBER_OF_HYPER_PARAM, 256)(enc_out)
        att_out = Dense(128, activation="tanh")(att_out)

        beta = Dense(1, activation="linear")(att_out[:, 0])
        gamma = Dense(1, activation="linear")(att_out[:, 1])
        mu = Dense(1, activation="linear")(att_out[:, 2])
        params = tf.concat([beta, gamma, mu], axis=-1)

        y_pred = Lambda(SIRD_layer)([inputs, params])

        model = Model(inputs=inputs, outputs=y_pred)

        return model

    def train(self, confirmed, recovered, deaths, epochs=10000, name="world"):
        S = (self.initN - confirmed) * 100 / self.initN
        I = (confirmed - recovered - deaths) * 100 / self.initN
        R = (recovered) * 100 / self.initN
        D = (deaths) * 100 / self.initN
        data = np.dstack([S, I, R, D])[0]

        data_generator = DataGenerator(data, data_len=self.day_lag, batch_size=1)

        def scheduler(epoch, lr):
            if epoch > 0 and epoch % 100 == 0:
                return lr*0.9
            else:
                return lr

        lr_schedule = LearningRateScheduler(scheduler)
        optimizer = Adam(learning_rate=1e-6)
        checkpoint = ModelCheckpoint(os.path.join('./ckpt', 'ckpt_%s_%d_{epoch:06d}.h5'%(name, self.day_lag)), period=5, save_best_only=True)
        early_stop = EarlyStopping(monitor="loss", patience=10)

        self.model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mean_absolute_error'])
        self.model.fit_generator(generator=data_generator, epochs=epochs, callbacks=[lr_schedule, checkpoint, early_stop])

        self.model.save_weights("%s_%d.h5"%(name, self.day_lag))

    def evaluate(self, confirmed, recovered, deaths):
        S = (self.initN - confirmed) * 100 / self.initN
        I = (confirmed - recovered - deaths) * 100 / self.initN
        R = (recovered) * 100 / self.initN
        D = (deaths) * 100 / self.initN
        data = np.dstack([S, I, R, D])[0]

        data_generator = DataGenerator(data, data_len=self.day_lag, batch_size=1)
        return self.model.evaluate_generator(data_generator, verbose=1)

    def predict(self, x, return_param=False):
        input_x = np.empty((1, self.day_lag, 4))
        x = np.array(x)
        scale_factor = 100

        S = ((self.initN - x[0]) / self.initN) * scale_factor
        I = ((x[0] - x[1] - x[2]) / self.initN) * scale_factor
        R = (x[1] / self.initN) * scale_factor
        D = (x[2] / self.initN) * scale_factor

        input_x = np.absolute(np.dstack([S, I, R, D]))

        result = self.model.predict(input_x)
        result = np.array(result, dtype=np.float64)

        if return_param:
            param_byu = self.estimator_model.predict(input_x)
            return (result/scale_factor)*self.initN, param_byu

        return (result/scale_factor)*self.initN

    def predict_estimator(self, x):
        input_x = np.empty((1, self.day_lag, 4))
        x = np.array(x)
        scale_factor = 100

        S = ((self.initN - x[0]) / self.initN) * scale_factor
        I = ((x[0] - x[1] - x[2]) / self.initN) * scale_factor
        R = (x[1] / self.initN) * scale_factor
        D = (x[2] / self.initN) * scale_factor

        input_x = np.dstack([S, I, R, D])

        result = self.estimator_model.predict(input_x)
        return result

if __name__ == '__main__':
    data_loader = DataLoader()
    becaked_model = BeCakedModel()
    start = 161
    end = 223

    ml_model.train(data[0][0:start], data[1][0:start], data[2][0:start], epochs=500)

    print(ml_model.evaluate(data[0][start-10:end], data[1][start-10:end], data[2][start-10:end]))
