import pandas
import tensorflow as tf
import numpy as np
from get_dataframe import data_structure

def collect_data(path):
    filename = path
    data_frame = pandas.read_csv(filename)
    # data_frame = tf.data.experimental.CsvDataset(filename, record_defaults)
    return data_frame

def prepare_data(data):
    array = np.array(data)
    ys = array[:, : -2]
    xs = array[1:]

    return xs, ys

    # ys = ys[:, :1]
    # print(ys)

    # xs = np.asarray(xs).astype(np.float32)
    # ys = np.asarray(ys).astype(np.float32)
    # ds = data_structure(data)
    # print(ds.element_spec)

def build_model(dataframe):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.summary()
    return model

def train_model(model, xs, ys):
    model.fit(xs, ys, epochs=500)

data = collect_data('/home/jm/Data/test_PS_20174392719_1491204439457_log.csv')
print(data.describe(''))
# print(model.predict([20]))
