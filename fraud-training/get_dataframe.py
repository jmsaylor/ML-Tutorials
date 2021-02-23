import tensorflow as tf
import numpy as np

def data_structure(dataframe, shuffle=True, batch_size=5):
    dataframe = dataframe.copy()
    labels = dataframe.pop('isFlaggedFraud')
    data_set = tf.data.Dataset.from_tensor_slices(dict(dataframe))


    data_set.batch(batch_size)
    return data_set