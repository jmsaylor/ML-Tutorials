import pandas
import tensorflow as tf
import numpy as np
from get_dataframe import data_structure

def prepare_data(path):
    filename = path
    data_frame = pandas.read_csv(
        filename,
        header=0,
        index_col=False,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        )


    data_frame['type'] = pandas.Categorical(data_frame['type'])

    print(data_frame.dtypes)

    labels = data_frame.pop('isFraud')
    features = data_frame

    data_set = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
    # print(data_set.element_spec)



    # ds = data_structure(data)
    # print(ds.element_spec)

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.summary()
    return model

def train_model(model, xs, ys):
    model.fit(xs, ys, epochs=500)

pandas.set_option("display.max_columns", 11)
prepare_data('/home/jm/Data/test_PS_20174392719_1491204439457_log.csv')
# prepare_dataset(data)
# print(model.predict([20]))
