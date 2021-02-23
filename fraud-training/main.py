import pandas
import tensorflow as tf
import numpy as np
from get_dataframe import data_structure

filename = '/home/jm/Data/test_PS_20174392719_1491204439457_log.csv'
data = pandas.read_csv(filename)
ds = data_structure(data)
ds.
print(data.head())


# data = np.array(data)
# data = data[1:]
# xs, ys = data[:, :-1], data[:, -1]
# xs = np.asarray(xs).astype(np.float32)
# ys = np.asarray(ys).astype(np.float32)
#
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[9,])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
#
# model.summary()
# model.fit(xs, ys, epochs=500)
#
# print(model.predict([20]))
