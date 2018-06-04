
import numpy as np
import matplotlib.pyplot as plt

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route

import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json

look_back = 30
sequence_length = 30

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])
    return np.array(dataX), np.array(dataY)

def create_data():
    path = "../annotations/hyang/video0/"
    loader = Loader(path)
    south = np.array([720, 1920])
    north = np.array([720, 0])
    route = Route(south, north)
    loader.make_obj_dict_by_route(route, True, 'Biker')
    postprocessor = PostProcessing(loader)
    global loader
    global postprocessor

    y = postprocessor.standardize(postprocessor.d, postprocessor.d)
    y = y.reshape(len(y), 1)
    # split into train and test sets
    train_size = int(len(y) * 0.67)
    train, test = y[0:train_size, :], y[train_size:len(y), :]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY

def train_model():
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    trainX, trainY, testX, testY = create_data()

    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    # serialize model to JSON
    model_json = model.to_json()
    with open("log/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("log/model.h5")
    print("Saved model to disk")


def load_model():
    # load json and create model
    json_file = open('log/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("log/model.h5")
    print("Loaded model from disk")
    return loaded_model


def evaluate_model():
    model = load_model()
    trainX, trainY, testX, testY = create_data()
    id = 32
    n_batches = len(postprocessor.filtered_dict[32]) // sequence_length
    plt.figure(figsize=(20, 10))
    for batch in range(n_batches):
        idx = sequence_length*batch
        d_input_sequence, d_output_sequence = postprocessor.get_batch_standardized(id, sequence_length + look_back - 1, idx)

        # convert lateral distance to a input sequence
        a = d_input_sequence[0:sequence_length]
        for i in range(1, look_back):
            a = np.vstack((a, d_input_sequence[i:sequence_length + i]))
        a = a.T
        a = a.reshape(a.shape[0], 1, a.shape[1])

        # do prediction
        y_hat = model.predict(a)
        y_hat = postprocessor.inv_standardize(y_hat, postprocessor.d)
        d_input_sequence = postprocessor.inv_standardize(d_input_sequence, postprocessor.d)
        d_output_sequence = postprocessor.inv_standardize(d_output_sequence, postprocessor.d)

        # convert to position coordinates
        y_hat_series = []
        for i, position in enumerate(
                postprocessor.filtered_dict[id][idx + look_back:idx + sequence_length + look_back]):
            s, d = ct.get_frenet_coord(postprocessor.route, position)
            y_hat_position, _ = ct.get_cart_coord_from_frenet(postprocessor.route, s, y_hat[i])
            y_hat_series.append(y_hat_position)
        y_hat_series = np.asarray(y_hat_series)

        plt.subplot(1, 2, 1)
        plt.cla()
        plt.imshow(loader.map)
        plt.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color='blue', label='trajectory')
        plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black', label='route')
        plt.plot(postprocessor.filtered_dict[id][idx:idx + sequence_length, 0],
                 postprocessor.filtered_dict[id][idx:idx + sequence_length, 1], color='red', label='input')
        plt.plot(postprocessor.filtered_dict[id][idx + sequence_length:idx + sequence_length*2, 0],
                 postprocessor.filtered_dict[id][idx + sequence_length:idx + sequence_length*2, 1], color='green', label='ground truth')
        plt.plot(y_hat_series[:, 0], y_hat_series[:, 1], color='orange', label='prediction')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.cla()
        plt.grid('On')
        plt.plot(np.arange(0, sequence_length), d_input_sequence[0:sequence_length], color='red', marker='+', label='input')
        plt.plot(np.arange(sequence_length, sequence_length * 2), d_output_sequence[0:sequence_length], color='green', marker='+', label='ground truth')
        plt.plot(np.arange(0, sequence_length)+look_back, y_hat, color='orange', marker='+', label='prediction')
        plt.xlim([0, 60])
        plt.ylim([-50, 50])
        plt.legend()

        plt.draw()
        plt.pause(0.1)
        plt.savefig('AB/' + str(id) + '/prediction/batch_' + str(batch))


evaluate_model()
#train_model()