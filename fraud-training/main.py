import pandas
import tensorflow as tf
tf.executing_eagerly()

from sklearn import preprocessing
import numpy as np
from get_dataframe import data_structure


def prepare_data(path):
    df = pandas.read_csv(path)
    df.pop('step'); df.pop('isFlaggedFraud')
    df.pop('nameOrig'); df.pop('nameDest')
    # print(df.head())

    data = {}

    columns = ['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
     'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']

    types = np.array(df.pop('type'))

    # #TODO: improve to one-hot encoding
    # label_encoder = preprocessing.LabelEncoder().fit(types)
    # types = label_encoder.transform(types)
    # data['type'] = np.array(types, dtype=np.int)

    numeric_data_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    numeric_data = []

    for name in numeric_data_cols:
        numeric_data.append(df.pop(name).values)

    amounts_scaler = preprocessing.MinMaxScaler().fit(numeric_data)
    scaled_numeric_data = amounts_scaler.transform(numeric_data)

    for index in range(len(numeric_data_cols)):
        data[numeric_data_cols[index]] = np.array(scaled_numeric_data[index], dtype=float)

    target = []
    isFraud = df.pop('isFraud').values
    for i in range(len(isFraud)):
        target.append([1, 1, 1, 1, isFraud[i]])
        # target.append([isFraud[i]])

    target = np.array(target)

    source_data = []
    for name in data:
        source_data.append(data[name])


    source_data = np.array(source_data)
    source_data = source_data.transpose()
    return source_data, target


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None , 5)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.summary()
    return model

def train_model(model, xs, ys):
    model.fit(xs, ys, epochs=500)

xs, ys = prepare_data('/home/jm/Data/test_PS_20174392719_1491204439457_log.csv')
print(xs)
model = build_model()
model.fit(xs, ys, epochs=5)
print(model.predict([[1,0,1,0,1]]))
