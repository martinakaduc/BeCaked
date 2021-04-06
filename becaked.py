from __future__ import division
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, LSTM, Dense, Activation, Concatenate, Add, Subtract, Multiply, Lambda, Reshape, Flatten, Dropout
import keras.backend as K
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import os
from data_utils import *
from utils import *
from generator import *

DAYS = 10

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
        input_raw = Input(shape=(day_lag, 4)) # S, I, R, D

        x = Lambda(case_diff)(input_raw)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(3, activation='linear')(x) # beta, gamma, muy
        x = Reshape((3,1))(x)
        y_pred = Lambda(SIRD_layer)([input_raw, x])
        model = Model(inputs=input_raw, outputs=y_pred)

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
        checkpoint = ModelCheckpoint(os.path.join('./ckpt', 'ckpt_%s_%d_{epoch:06d}.h5'%(name, self.day_lag)), period=500)
        early_stop = EarlyStopping(monitor="loss", patience=100)

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
