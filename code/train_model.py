from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

from keras.models import model_from_json

from datetime import datetime
# load data
# def parse(x):
# 	return datetime.strptime(x, '%Y %m %d %H')
# dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# # manually specify column names
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# # mark all NA values with 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours
# dataset = dataset[24:]
# # summarize first 5 rows
# print(dataset.head(5))
# # save to file
# dataset.to_csv('pollution.csv')


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

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route

import numpy as np

path = "../annotations/hyang/video0/"
loader = Loader(path)
south = np.array([720, 1920])
north = np.array([720, 0])
west = np.array([720*2, 1920/2])
route = Route(south, west)
loader.make_obj_dict_by_route(route, True, 'Biker')
postprocessor = PostProcessing(loader)

raw = DataFrame()
#raw['x'] = [x for x in postprocessor.x]
#raw['y'] = [x for x in postprocessor.y]
raw['xdot'] = [x for x in postprocessor.dx]
raw['ydot'] = [x for x in postprocessor.dy]
# raw['xddot'] = [x for x in postprocessor.ddx]
# raw['yddot'] = [x for x in postprocessor.ddy]
values = raw.values
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
data = series_to_supervised(values, 30, 30)
print(data)
values = data.values
n_train_frames = int(len(data) * 0.67)
train = values[:n_train_frames, :]
test = values[n_train_frames:, :]
n_hours = 30
n_features = 2

n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, n_obs:]#train[:, train.shape[1]-n_features: train.shape[1]]
test_X, test_y = test[:, :n_obs], test[:, n_obs:] #test[:, test.shape[1]-n_features: test.shape[1]]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

train_y = train_y.reshape((train_y.shape[0], n_hours, n_features))
test_y = test_y.reshape((test_y.shape[0], n_hours, n_features))
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

def train_model():
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)

    # serialize model to JSON
    model_json = model.to_json()
    with open("log_xy/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("log_xy/model.h5")
    print("Saved model to disk")

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.grid('On')
    pyplot.legend()
    pyplot.savefig('log_xy/loss')
    pyplot.show()

    return model

def evaluate_model():
    # load json and create model
    json_file = open('log_xy/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("log_xy/model.h5")
    print("Loaded model from disk")
    return loaded_model

model = train_model()
test_X_in = test_X
test_y_in = test_y

yhat = model.predict(test_X_in)
pred_series = np.empty([yhat.shape[0], yhat.shape[1], yhat.shape[2]])
input_series = np.empty([test_X_in.shape[0], test_X_in.shape[1], test_X_in.shape[2]])
output_series = np.empty([test_y_in.shape[0], test_y_in.shape[1], test_y_in.shape[2]])
colors = pyplot.cm.gist_ncar(np.linspace(.1, .9, np.max(postprocessor.id+1)))

tot_error = 0
for t in range(pred_series.shape[0]):
    pred_series[t] = yhat[t]#scaler.inverse_transform(yhat[t])
    input_series[t] = test_X_in[t]#scaler.inverse_transform(test_X_in[t])
    output_series[t] = test_y_in[t]#scaler.inverse_transform(test_y_in[t])
    id = int(postprocessor.id[n_train_frames + t])
    # dx = np.hstack((np.array([0]), np.diff(postprocessor.x[n_train_frames + t: n_train_frames + t + n_hours])))
    # dy = np.hstack((np.array([0]), np.diff(postprocessor.y[n_train_frames + t: n_train_frames + t + n_hours])))
    # dxy = np.vstack((dx, dy))
    # output_series[t] = np.vstack((dx, dy)).T
    #
    # dx = np.hstack((np.array([0]), np.diff(postprocessor.x[n_train_frames + t - n_hours: n_train_frames + t])))
    # dy = np.hstack((np.array([0]), np.diff(postprocessor.y[n_train_frames + t - n_hours: n_train_frames + t])))
    # dxy = np.vstack((dx, dy))
    # input_series[t] = np.vstack((dx, dy)).T

    #pred_series[t] = model.predict(input_series[t].reshape(1, test_X_in.shape[1], test_X_in.shape[2]))  # scaler.inverse_transform(yhat[t])

    pyplot.cla()
    pyplot.imshow(loader.map)

    pyplot.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color=colors[id])
    x_start = postprocessor.x[n_train_frames + t + n_hours]
    y_start = postprocessor.y[n_train_frames + t + n_hours]

    x_start_input = postprocessor.x[n_train_frames + t - n_hours + n_hours]
    y_start_input = postprocessor.y[n_train_frames + t - n_hours + n_hours]
    pyplot.quiver(x_start, y_start, pred_series[t, 0, 0], -pred_series[t, 0, 1], color='red')
    pyplot.plot(np.cumsum(pred_series[t, :, 0]) + x_start, np.cumsum(pred_series[t, :, 1])+y_start, label='y_hat', color='orange')
    pyplot.plot(np.cumsum(input_series[t, :, 0]) + x_start_input, np.cumsum(input_series[t, :, 1])+y_start_input, label='x', color='red')
    pyplot.plot(np.cumsum(output_series[t, :, 0]) + x_start, np.cumsum(output_series[t, :, 1])+y_start, label='y', color='green')
    pyplot.legend()
    pyplot.draw()
    pyplot.pause(0.001)
    print('id= ' + str(id))
    rmse = sqrt(mean_squared_error(pred_series[t], output_series[t]))
    print('Test RMSE: %.3f' % rmse)
    tot_error+=rmse

    #pyplot.show()

print('Test total error: %.3f' % tot_error)

# calculate RMSE
