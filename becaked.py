from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Reshape, Flatten, Dropout, GRU, BatchNormalization, Dot, Concatenate, Activation
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import initializers
import os
from data_utils import *
from utils import *
from generator import *

DAYS = 10
NUMBER_OF_HYPER_PARAM = 7
NUM_HIDDEN = 128

class Attention(Layer):
    def __init__(self, units=128, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28, philipperemy.
        """
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def SIRD_layer(tensors):
    input_raw, x = tensors
    #input_raw: [S,E,I,R,D,beta]
    #x: [gamma,muy,eta,xi,theta,sigma,beta_bar]


    # S = S - beta*S*E + xi*R - theta*S
    S = tf.add(
                tf.subtract(
                    input_raw[:,:,0],
                    tf.multiply(tf.multiply(x[:,6],input_raw[:,:,0]),input_raw[:,:,1])
                    ),
                tf.subtract(
                    tf.multiply(x[:,3],input_raw[:,:,3]),
                    tf.multiply(x[:,4],input_raw[:,:,0])
                    )
            )


    # E = E + beta*S*E - eta*E
    E = tf.add(
            input_raw[:,:,1],
            tf.subtract(
                tf.multiply(tf.multiply(x[:,6],input_raw[:,:,0]),input_raw[:,:,1]),
                tf.multiply(x[:,2],input_raw[:,:,1])
            )
        )

    # I = I + eta*E - gamma*I - muy*I + theta*S
    I = tf.add(
        tf.add(input_raw[:,:,2],tf.multiply(x[:,4],input_raw[:,:,0])),
        tf.subtract(
            tf.subtract(
                tf.multiply(x[:,2],input_raw[:,:,1]),
                tf.multiply(x[:,0],input_raw[:,:,2])
                ),
            tf.multiply(x[:,1],input_raw[:,:,2]),
        )
    )

    # R = R + gamma*I - xi*R
    R = tf.add(
        input_raw[:,:,3],
        tf.subtract(
            tf.multiply(x[:,0],input_raw[:,:,2]),
            tf.multiply(x[:,3],input_raw[:,:,3])
        )
    )

    # D = D + muy*I
    D = tf.add(
        input_raw[:,:,4],
        tf.multiply(x[:,1],input_raw[:,:,2])
    )

    # beta = beta - theta*(beta - beta_bar) + sigma*dW/dt
    # dW/dt ~ N(0,1)

    rand = tf.random.normal([1,DAYS],0,1)
    beta = tf.subtract(
        input_raw[:,:,5],
        tf.subtract(
            tf.multiply(
                x[:,4],
                tf.subtract(input_raw[:,:,5], x[:,6])
            ),
            tf.multiply(x[:,5],rand)
        )
    )

    # beta = tf.sigmoid(tf.multiply(input_raw[:,:,5], x[:,6]))

    N = tf.add(
        input_raw[:,:,0],
        tf.add(input_raw[:,:,1],input_raw[:,:,3])
    )

    out = tf.stack([S, E, I, R, D, beta, N], axis=-1)
    return out

def case_diff(tensor):
    return tf.subtract(tensor[:,1:], tensor[:,:-1])

class BeCakedModel():
    def __init__(self, population=8993082, day_lag=DAYS):
        self.initN = population
        self.day_lag = day_lag
        self.model = self.build_model(day_lag)

        # if os.path.exists("models/world_%d.h5"%day_lag):
            # self.load_weights("models/world_%d.h5"%day_lag)

        self.model.summary()
        self.estimator_model = Model(inputs=self.model.input,
                                        outputs=self.model.layers[-2].output)

    def update_population(self, population):
        self.initN = population

    def reset_population(self):
        self.initN = 8993082

    def load_weights(self, path):
        print("Loading saved model at %s"%path)
        self.model.load_weights(path)
        self.estimator_model = Model(inputs=self.model.input,
                                        outputs=self.model.layers[-2].output)

    def build_model(self, day_lag, num_hidden=NUM_HIDDEN):
        inputs = Input(shape=(day_lag, 7)) # S, E, I, R, D, beta, N

        acce = Lambda(case_diff)(inputs[:,:,:-2])
        enc_in, enc_state = GRU(num_hidden,
                          # Return the sequence and state
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')(acce)
        enc_out, _ = GRU(num_hidden,
                          # Return the sequence and state
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')(enc_in, initial_state=enc_state)

        att_out = Attention(2*num_hidden)(enc_out)
        att_out = Flatten()(att_out)

        dn_1 = Dense(num_hidden, activation="tanh")(att_out)
        # dl_1 = Dense(num_hidden, activation="linear")(att_out)
        # d1 = Concatenate()([dn_1, dl_1])

        params = Dense(NUMBER_OF_HYPER_PARAM, activation="tanh")(dn_1)  # gamma, muy, eta, xi, theta, sigma, beta_bar

        y_pred = Lambda(SIRD_layer)([inputs, params])

        model = Model(inputs=inputs, outputs=y_pred)

        return model

    def my_mean_squared_error(self,y_true, y_pred):
        #ignore beta
        mse = MeanSquaredError()
        return mse(y_true[:,:,2:5],y_pred[:,:,2:5])

    def train(self, exposed, infectious, recovered, deaths, beta, N, epochs=1000, name="world"):
        self.update_population(N[-1])

        S = (N - exposed - infectious - recovered - deaths) * 100 / N
        E = (exposed) * 100 / N
        I = (infectious) * 100 / N
        R = (recovered) * 100 / N
        D = (deaths) * 100 / N
        N = np.ones_like(S)*100.
        beta = np.ones_like(D)
        data = np.dstack([S, E, I, R, D, beta, N])[0]

        data_generator = DataGenerator(data, data_len=self.day_lag, batch_size=1)

        def scheduler(epoch, lr):
            if epoch > 0 and epoch % 100 == 0:
                return lr*0.9
            else:
                return lr

        lr_schedule = LearningRateScheduler(scheduler)
        optimizer = Adam(learning_rate=1e-5)
        # checkpoint = ModelCheckpoint(os.path.join('./ckpt', 'ckpt_%s_%d_{epoch:06d}.h5'%(name, self.day_lag)), save_freq=10)
        early_stop = EarlyStopping(monitor="loss", patience=50)

        self.model.compile(optimizer=optimizer, loss=self.my_mean_squared_error, metrics=['mean_absolute_error'])
        self.model.fit(data_generator, epochs=epochs, callbacks=[lr_schedule, early_stop], verbose=0)

        # self.model.save_weights("models/%s_%d.h5"%(name, self.day_lag))

    def evaluate(self, exposed, infectious, recovered, deaths):
        S = (self.initN - exposed - infectious - recovered - deaths) * 100 / self.initN
        E = (exposed) * 100 / self.initN
        I = (infectious) * 100 / self.initN
        R = (recovered) * 100 / self.initN
        D = (deaths) * 100 / self.initN
        beta = np.ones_like(D)
        data = np.dstack([S, E, I, R, D, beta])[0]

        data_generator = DataGenerator(data, data_len=self.day_lag, batch_size=1)
        return self.model.evaluate_generator(data_generator, verbose=1)

    def predict(self, x, return_param=False):
        #x : [E,I,R,D]
        input_x = np.empty((1, self.day_lag, 5))
        x = np.array(x)
        scale_factor = 100

        S = ((self.initN - x[0] - x[1] - x[2] - x[3]) / self.initN) * scale_factor
        E = (x[0] / self.initN) * scale_factor
        I = (x[1] / self.initN) * scale_factor
        R = (x[2] / self.initN) * scale_factor
        D = (x[3] / self.initN) * scale_factor
        N = np.ones_like(S)*100.
        beta = np.ones_like(D)

        input_x = np.absolute(np.dstack([S, E, I, R, D, beta,N]))

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

        S = ((self.initN - x[0] - x[1] - x[2] - x[3]) / self.initN) * scale_factor
        E = (x[0] / self.initN) * scale_factor
        I = (x[1] / self.initN) * scale_factor
        R = (x[2] / self.initN) * scale_factor
        D = (x[3] / self.initN) * scale_factor
        beta = np.zeros_like(D)

        input_x = np.dstack([S, E, I, R, D, beta])

        result = self.estimator_model.predict(input_x)
        return result

if __name__ == '__main__':
    data_loader = DataLoader()
    becaked_model = BeCakedModel()
    start = 161
    end = 223
    data = data_loader.get_data_world_series()

    becaked_model.train(data[0][:start], data[1][:start], data[2][:start], epochs=500)

    print(becaked_model.evaluate(data[0][start-10:end], data[1][start-10:end], data[2][start-10:end]))
