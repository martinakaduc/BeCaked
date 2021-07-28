from __future__ import division
import numpy as np
from keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, data, data_len=10, batch_size=32):
        self.data = data
        self.data_len = data_len
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int((len(self.data) - self.data_len) / self.batch_size) + 1

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(indexes)

        return X[:,:self.data_len], X[:,1:]

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.data) - self.data_len)

    def _generate_X(self, list_index):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, self.data_len+1, 4))

        # Generate data
        for i, ID in enumerate(list_index):
            data_seq = self.data[ID:ID+self.data_len+1]
            X[i] = data_seq

        return X
