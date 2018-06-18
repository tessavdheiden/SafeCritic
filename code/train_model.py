from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import LSTM, Bidirectional
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import optimizers

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route

import numpy as np


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

path = "../annotations/hyang/video5/"
loader = Loader(path)
south = np.array([720, 1920])
north = np.array([720, 0])
east = np.array([720*2, 1920/2])
route1 = Route(north, south)
route2 = Route(east, south)
#loader.make_obj_dict_by_route(True, 'Biker', compute_all_routes=False)


postprocessor = PostProcessing(loader)

raw = DataFrame()
#raw['x'] = [x for x in postprocessor.x]
#raw['y'] = [x for x in postprocessor.y]
raw['xdot'] = [x for x in postprocessor.dx]
raw['ydot'] = [x for x in postprocessor.dy]
# raw['xdot'] = [x for x in postprocessor.ddx]
# raw['ydot'] = [x for x in postprocessor.ddy]

# raw['x0'] = [x for x in postprocessor.x0]
# raw['x1'] = [x for x in postprocessor.x1]
# raw['x2'] = [x for x in postprocessor.x2]
# raw['x3'] = [x for x in postprocessor.x3]
# raw['x4'] = [x for x in postprocessor.x4]
# raw['x5'] = [x for x in postprocessor.x5]
# raw['x6'] = [x for x in postprocessor.x6]
# raw['x7'] = [x for x in postprocessor.x7]
# raw['x8'] = [x for x in postprocessor.x8]
# raw['x9'] = [x for x in postprocessor.x9]
# raw['x10'] = [x for x in postprocessor.x10]
# raw['x11'] = [x for x in postprocessor.x11]
# raw['x12'] = [x for x in postprocessor.x12]
# raw['x13'] = [x for x in postprocessor.x13]

N_SAMPLES = 30
N_INPUT_FEATURES = 2 #+ 7 + 7
N_OUTPUT_FEATURES = 2
N_OBS = N_SAMPLES * N_INPUT_FEATURES
N_PRED = N_SAMPLES * N_OUTPUT_FEATURES
folder_name = 'log_bidirectional_no_dynamic_no_static'
mode = 'evaluate_train'

values = raw.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
data = series_to_supervised(scaled, N_SAMPLES, N_SAMPLES)

values = data.values
n_train_frames = int(len(data) * 0.87)
train = values[:n_train_frames, :]
test = values[n_train_frames:, :]
train_X = train[:, :N_OBS]
test_X = test[:, :N_OBS]

train_y = np.zeros((n_train_frames, N_PRED))
test_y = np.zeros((values.shape[0] - n_train_frames, N_PRED))
for column in range(N_PRED // N_OUTPUT_FEATURES):
    start_column = N_OBS+column*N_INPUT_FEATURES
    start_column_in = column*N_OUTPUT_FEATURES
    print('start_column=',start_column)
    print('start_column_in=', start_column_in)
    train_y[:, start_column_in:start_column_in+2] = train[:, start_column:start_column+2]
    test_y[:, start_column_in:start_column_in+2] = test[:, start_column:start_column+2]

train_X = train_X.reshape((train_X.shape[0], N_SAMPLES, N_INPUT_FEATURES))# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], N_SAMPLES, N_INPUT_FEATURES))

train_y = train_y.reshape((train_y.shape[0], N_SAMPLES, N_OUTPUT_FEATURES))
test_y = test_y.reshape((test_y.shape[0], N_SAMPLES, N_OUTPUT_FEATURES))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def plot_input():
    # specify columns to plot
    groups = [0, 1]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(raw.values[:, group])
        pyplot.title(raw.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

def my_model():

    # design network
    #model = Sequential()
    input_sequences = Input(shape=(train_X.shape[1], train_X.shape[2]))
    layer1 = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat')(input_sequences)
    layer2 = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat')(layer1)
    output_sequence = TimeDistributed(Dense(N_OUTPUT_FEATURES, activation='relu'))(layer2)
    model = Model(inputs=input_sequences, outputs=output_sequence)
    model.compile(loss='mae', optimizer='adam')
    return model

def train_model(reset=False, early_stopping=False, n_losses=10):
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2,
                              write_graph=True, write_images=False)
    model = my_model()
    # fit network
    loss, val_loss = np.array([]), np.array([])
    if reset:
        ids = np.unique(postprocessor.id[:n_train_frames])
        val_ids = np.unique(postprocessor.id[n_train_frames:])
        for i in range(len(ids)):
            frames = postprocessor.id[:n_train_frames] == ids[i]
            val_frames = postprocessor.id[n_train_frames + 2*N_SAMPLES - 1:] == val_ids[i % len(val_ids)]
            trainX = train_X[frames]
            trainY = train_y[frames]
            testX = test_X[val_frames]
            testY = test_y[val_frames]
            history = model.fit(trainX, trainY, epochs=150//len(ids), batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
            loss = np.concatenate((loss,history.history['loss']))
            val_loss = np.concatenate((val_loss,history.history['val_loss']))
            model.reset_states()
    elif early_stopping:
        for e in range(50):
            history = model.fit(train_X, train_y, epochs=1, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False, callbacks=[tensorboard])
            loss = np.concatenate((loss, history.history['loss']))
            val_loss = np.concatenate((val_loss, history.history['val_loss']))
            if ((np.abs(np.diff(val_loss[-n_losses:])) < 0.001).all() or ((np.diff(val_loss[-n_losses:])) > 0.01).all()) and e > n_losses: # no weight update or increasing loss
                print(np.diff(val_loss[-n_losses:]))
                break
    else:
        history = model.fit(train_X, train_y, epochs=20, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False, callbacks=[tensorboard])

        loss = history.history['loss']
        val_loss = history.history['val_loss']

    # serialize model to JSON
    model_json = model.to_json()
    with open(folder_name + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(folder_name + "/model.h5")
    print("Saved model to disk")

    # plot history
    pyplot.plot(loss, label='train')
    pyplot.plot(val_loss, label='test')
    pyplot.grid('On')
    pyplot.legend()
    pyplot.savefig(folder_name + '/loss')
    #pyplot.show()

    return model

def evaluate_model():
    # load json and create model
    json_file = open(folder_name + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(folder_name + "/model.h5")
    print("Loaded model from disk")
    return loaded_model

if mode == 'train':
    model = train_model()
else:
    model = evaluate_model()

if mode == 'evaluate_train':
    test_X_in = train_X
    test_y_in = train_y
    n_train_frames = 0
else:
    test_X_in = test_X
    test_y_in = test_y

yhat = model.predict(test_X_in)
pred_series = np.empty([test_y_in.shape[0], test_y_in.shape[1], test_y_in.shape[2]])

pred_series_baseline = np.empty([test_y_in.shape[0], test_y_in.shape[1], 2])
#output_series_baseline = np.empty([test_y_in.shape[0], test_y_in.shape[1], 2])
input_series = np.empty([test_X_in.shape[0], test_X_in.shape[1], test_X_in.shape[2]])
output_series = np.empty([test_y_in.shape[0], test_y_in.shape[1], test_y_in.shape[2]])
colors = pyplot.cm.gist_ncar(np.linspace(.1, .9, np.max(postprocessor.id+1)))

tot_error1 = 0
tot_error2 = 0
counter=0
pyplot.figure(figsize=(20, 20))

n_buckets = 3
errors = np.zeros((output_series.shape[0], n_buckets, 2))

benchmark = pd.DataFrame(np.zeros((output_series.shape[0], n_buckets*2)), columns=['lstm3', 'poly3','lstm6', 'poly6','lstm9', 'poly9'])

for t in range(0, output_series.shape[0], 10):
    if True:
        pred_series[t] = scaler.inverse_transform(np.hstack((yhat[t], test_X_in[t][:, 2:])))[:, 0:2]
        input_series[t] = scaler.inverse_transform(test_X_in[t])
        output_series[t] = scaler.inverse_transform(np.hstack((test_y_in[t], test_X_in[t][:, 2:])))[:, 0:2]
    else:
        pred_series[t] = yhat[t]
        input_series[t] = test_X_in[t]
        output_series[t] = test_y_in[t]

    id = int(postprocessor.id[n_train_frames + t])

    trajectory = np.squeeze(np.asarray(list(postprocessor.raw_dict[id].trajectory.values())))[10:-10]
    x_start = postprocessor.x[n_train_frames + t + N_SAMPLES]
    y_start = postprocessor.y[n_train_frames + t + N_SAMPLES]

    tx = np.arange(0, N_SAMPLES)
    p3x = np.poly1d(np.polyfit(tx, input_series[t, :, 0], 3))
    dx = p3x(tx)

    p3y = np.poly1d(np.polyfit(tx, input_series[t, :, 1], 3))
    dy = p3y(tx)
    pred_series_baseline[t] = np.vstack((dx, dy)).T

    pyplot.cla()
    pyplot.imshow(loader.map)
    pyplot.plot(trajectory[:, 0], trajectory[:, 1], color='black', label=str(id), linestyle='--', linewidth=1,
                alpha=0.5)
    x_start_input = postprocessor.x[n_train_frames + t]
    y_start_input = postprocessor.y[n_train_frames + t]

    pyplot.plot(np.cumsum(input_series[t, :, 0]) + x_start_input, np.cumsum(input_series[t, :, 1])+y_start_input, label='x',
                color='red', linestyle='--', linewidth=1)
    pyplot.plot(np.cumsum(output_series[t, :, 0]) + x_start, np.cumsum(output_series[t, :, 1])+y_start, label='ground truth',
                color='green', linestyle='--', linewidth=1)
    pyplot.plot(np.cumsum(pred_series_baseline[t, :, 0]) + x_start, np.cumsum(pred_series_baseline[t, :, 1]) + y_start, label='poly3',
                color='blue', linestyle='--', linewidth=1)
    pyplot.plot(np.cumsum(pred_series[t, :, 0]) + x_start, np.cumsum(pred_series[t, :, 1]) + y_start, label='lstm',
                color='purple', linestyle='--', linewidth=1)

    pyplot.legend()
    if (pred_series_baseline[t] == None).any() or (output_series[t] == None).any():
        print(pred_series_baseline[t])

    rmse1 = sqrt(mean_squared_error(pred_series_baseline[t], output_series[t]))
    rmse2 = sqrt(mean_squared_error(pred_series[t], output_series[t]))
    pyplot.xlabel('Test RMSE poly3 = %.3f lstm = %.3f Test total RMSE poly3 = %.3f lstm = %.3f' % (rmse1, rmse2, tot_error1, tot_error2))
    #benchmark.lstm(t)

    # for i in range(n_buckets):
    #     start = i*n_buckets*(n_hours // n_buckets)
    #     end = start + (n_hours // n_buckets)
    #     errors[t][i][0] = sqrt(mean_squared_error(pred_series_baseline[t][end], output_series_baseline[t][end]))
    #     errors[t][i][1] = sqrt(mean_squared_error(pred_series[t][end], pred_series[t][end]))

    #pyplot.boxplot(errors[:][:][0])

    tot_error1+=rmse1
    tot_error2 += rmse2
    pyplot.draw()
    #pyplot.pause(0.001)
    pyplot.savefig(folder_name+'/t_' + str(counter))
    counter+=1



    #pyplot.show()

print('Test total error poly: %.3f' % tot_error1)
print('Test total error lstm: %.3f' % tot_error2)

# calculate RMSE
