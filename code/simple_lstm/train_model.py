from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from scipy.interpolate import spline
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.layers import LSTM, RepeatVector, Conv1D, Conv2D, merge, concatenate, Permute, Reshape, Flatten
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf
from keras import optimizers

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route
import random
import numpy as np
import imageio

N_GRID_CELLS = 15
SINGLE_ATTENTION_VECTOR = False
N_SAMPLES = 30
N_INPUT_FEATURES = 2 + N_GRID_CELLS# + N_GRID_CELLS# + 1
N_OUTPUT_FEATURES = 2
N_OBS = N_SAMPLES * N_INPUT_FEATURES
N_PRED = N_SAMPLES * N_OUTPUT_FEATURES
SPLIT = 0.98

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


def split_to_test_train(data, n_train_frames, n_input_features):
    n_obs = N_SAMPLES * n_input_features
    values = data.values
    #n_train_frames = int(len(data) * split)
    train = values[:n_train_frames, :]
    test = values[n_train_frames:, :]
    train_X = train[:, :n_obs]
    test_X = test[:, :n_obs]
    train_X = train_X.reshape((train_X.shape[0], N_SAMPLES, n_input_features))  # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], N_SAMPLES, n_input_features))

    # y
    train_y = np.zeros((n_train_frames, N_PRED))
    test_y = np.zeros((values.shape[0] - n_train_frames, N_PRED))
    for column in range(N_PRED // N_OUTPUT_FEATURES):
        start_column = n_obs + column * n_input_features
        start_column_in = column * N_OUTPUT_FEATURES
        train_y[:, start_column_in:start_column_in + N_OUTPUT_FEATURES] = train[:, start_column:start_column + N_OUTPUT_FEATURES]
        test_y[:, start_column_in:start_column_in + N_OUTPUT_FEATURES] = test[:, start_column:start_column + N_OUTPUT_FEATURES]

    train_y = train_y.reshape((train_y.shape[0], N_SAMPLES, N_OUTPUT_FEATURES))
    test_y = test_y.reshape((test_y.shape[0], N_SAMPLES, N_OUTPUT_FEATURES))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def make_df_from_postprocessor(postprocessor, n_input_features):
    raw = DataFrame()
    raw['xdot'] = [x for x in postprocessor.dx]
    raw['ydot'] = [x for x in postprocessor.dy]
    if n_input_features > 2 + N_GRID_CELLS:
        for i in range(2*N_GRID_CELLS):
            x_ = getattr(postprocessor, 'x' + str(i))
            raw['x' + str(i)] = [x for x in x_]
    elif n_input_features > 2:
        for i in range(N_GRID_CELLS):
            x_ = getattr(postprocessor, 'x' + str(i))
            raw['x' + str(i)] = [x for x in x_]

    raw['frame'] = [x for x in postprocessor.frame]
    return raw

def make_df_from_postprocessor_within_selection(postprocessor, selection, n_input_features):
    raw = DataFrame()
    raw['xdot'] = [x for x in postprocessor.dx[selection]]
    raw['ydot'] = [x for x in postprocessor.dy[selection]]

    if n_input_features > 2 + N_GRID_CELLS:
        for i in range(2*N_GRID_CELLS):
            x_ = getattr(postprocessor, 'x' + str(i))
            raw['x' + str(i)] = [x for x in x_[selection]]
    elif n_input_features > 2:
        for i in range(N_GRID_CELLS):
            x_ = getattr(postprocessor, 'x' + str(i))
            raw['x' + str(i)] = [x for x in x_[selection]]

    raw['frame'] = [x for x in postprocessor.frame[selection]]
    return raw

def plot_results(path_list, label_list):
    colors = pyplot.cm.gist_ncar(np.linspace(.1, .9, len(path_list)))
    for i, path in enumerate(path_list):
        locals()['path{}'.format(i)] = path
        locals()['loss{}'.format(i)] = np.load(locals()['path{}'.format(i)] + '/loss.npy')
        locals()['val_loss{}'.format(i)] = np.load(locals()['path{}'.format(i)] + '/val_loss.npy')
        x = locals()['val_loss{}'.format(i)]
        pyplot.plot(x, label=label_list[i], linestyle='--', color=colors[i], linewidth=1)
        pyplot.plot(locals()['val_loss{}'.format(i)], label=label_list[i], linestyle=':', color=colors[i], linewidth=1)

    pyplot.grid('On')
    pyplot.legend()
    pyplot.show()


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, N_SAMPLES))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(N_SAMPLES, activation='sigmoid')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = merge([inputs, a_probs], mode='mul')
    return output_attention_mul


def model_attention_applied_before_lstm():

    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    if N_INPUT_FEATURES > 2:
        input_sequences_1 = crop(2, 0, 2)(inputs)
        input_sequences_2 = crop(2, 2, 2 + N_GRID_CELLS)(inputs)
        attention_mul1 = attention_3d_block(input_sequences_1)
        attention_mul2 = attention_3d_block(input_sequences_2)
        attention_mul = concatenate([attention_mul1, attention_mul2])
    else:
        attention_mul = attention_3d_block(inputs)

    lstm_units = 50
    attention_mul1 = LSTM(lstm_units, return_sequences=True)(attention_mul)
    attention_mul2 = LSTM(lstm_units, return_sequences=True)(attention_mul1)
    attention_mul = LSTM(lstm_units, return_sequences=True)(attention_mul2)
    output = TimeDistributed(Dense(N_OUTPUT_FEATURES, activation='sigmoid'))(attention_mul)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mae', optimizer='adam')
    return model


def model_attention_applied_after_lstm():
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm_units = 50
    lstm_out1 = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out2 = LSTM(lstm_units, return_sequences=True)(lstm_out1)
    lstm_out = LSTM(lstm_units, return_sequences=True)(lstm_out2)
    attention_mul = attention_3d_block(lstm_out)
    #attention_mul = Flatten()(attention_mul)
    output = TimeDistributed(Dense(N_OUTPUT_FEATURES, activation='sigmoid'))(attention_mul)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mae', optimizer='adam')
    return model

def my_model():
    input_sequences = Input(shape=(train_X.shape[1], train_X.shape[2]))

    if N_INPUT_FEATURES > 2 + N_GRID_CELLS:
        input_sequences_1 = crop(2, 0, 2)(input_sequences)
        input_sequences_2 = crop(2, 2, 2 + N_GRID_CELLS)(input_sequences)
        input_sequences_3 = crop(2, 2 + N_GRID_CELLS, 2 + N_GRID_CELLS*2)(input_sequences)

        layer1 = LSTM(50, return_sequences=True)(input_sequences_1)
        layer2 = LSTM(1, return_sequences=True)(input_sequences_2)
        layer3 = LSTM(1, return_sequences=True)(input_sequences_3)
        concatenated = concatenate([layer1, layer2, layer3])
    elif N_INPUT_FEATURES > 2:
        input_sequences_1 = crop(2, 0, 2)(input_sequences)
        input_sequences_2 = crop(2, 2, 2 + N_GRID_CELLS)(input_sequences)
        layer1 = LSTM(50, return_sequences=True)(input_sequences_1)
        layer2 = LSTM(50, return_sequences=True)(input_sequences_2)
        concatenated = concatenate([layer1, layer2])
    else:
        input_sequences_1 = crop(2, 0, 2)(input_sequences)
        layer1 = LSTM(50, return_sequences=True)(input_sequences_1)
        concatenated = layer1

    intermediate_forward = LSTM(50, return_sequences=True)(concatenated)
    intermediate_backward = LSTM(50, return_sequences=True)(intermediate_forward)
    layer2 = LSTM(50, return_sequences=True)(intermediate_backward)
    output_sequence = TimeDistributed(Dense(N_OUTPUT_FEATURES, activation='sigmoid'))(layer2)

    model = Model(inputs=input_sequences, outputs=output_sequence)
    model.compile(loss='mae', optimizer='adam')
    return model


def train_model(early_stopping=False, n_losses=10):
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2,
                              write_graph=True, write_images=False)
    model = model_attention_applied_before_lstm()
    # fit network
    loss, val_loss = np.array([]), np.array([])
    if reset:
        for e in range(epochs):
            print('Epch: %i' % (e))
            random.seed(e)
            ids = np.random.choice(np.unique(postprocessor.id[:n_train_frames]),
                                   len(np.unique(postprocessor.id[:n_train_frames])))
            ids_val = np.random.choice(np.unique(postprocessor.id[n_train_frames:]),
                                       len(np.unique(postprocessor.id[:n_train_frames])))
            for i, id in enumerate(ids):
                frames = postprocessor.id == id
                frames_val = postprocessor.id == ids_val[i]
                raw = make_df_from_postprocessor_within_selection(postprocessor, frames, N_INPUT_FEATURES)
                raw_val = make_df_from_postprocessor_within_selection(postprocessor, frames_val, N_INPUT_FEATURES)
                print('\nTrain ID %i with %i frames of total %i' % (id, frames[frames == True].shape[0], frames[frames == False].shape[0]))
                print('Test ID %i with %i frames of total %i' % (ids_val[i], frames_val[frames_val == True].shape[0], frames_val[frames_val == False].shape[0]))
                scaled = scaler.transform(raw.values[:, :N_INPUT_FEATURES])
                scaled_val = scaler.transform(raw_val.values[:, :N_INPUT_FEATURES])
                data = series_to_supervised(scaled, N_SAMPLES, N_SAMPLES)
                data_val = series_to_supervised(scaled_val, N_SAMPLES, N_SAMPLES)
                trainX, trainY, _, _ = split_to_test_train(data, data.shape[0], N_INPUT_FEATURES)
                testX, testY, _, _ = split_to_test_train(data_val, data_val.shape[0], N_INPUT_FEATURES)

                if testX.shape[0] > 2 * N_SAMPLES and trainX.shape[0] > 2 * N_SAMPLES:  # will occur when there are not enough samples than 2*seq_length
                    history = model.fit(trainX, trainY, epochs=1, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
                    loss = np.concatenate((loss,history.history['loss']))
                    val_loss = np.concatenate((val_loss,history.history['val_loss']))
                model.reset_states()
    elif early_stopping:
        for e in range(50):
            print('Epoch: ', str(e))
            history = model.fit(train_X, train_y, epochs=1, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False, callbacks=[tensorboard])
            loss = np.concatenate((loss, history.history['loss']))
            val_loss = np.concatenate((val_loss, history.history['val_loss']))
            if ((np.abs(np.diff(val_loss[-n_losses:])) < 0.001).all() or ((np.diff(val_loss[-n_losses:])) > 0.001).all()) and e > n_losses: # no weight update or increasing loss
                print(np.diff(val_loss[-n_losses:]))
                break
    else:
        history = model.fit(train_X, train_y, epochs=15, batch_size=72, validation_data=(test_X, test_y), verbose=2,
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
    np.save(folder_name + '/loss.npy', loss)
    np.save(folder_name + '/val_loss.npy', val_loss)
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


def calc_rms(predicted, actual, idx):
    v_pred = np.hstack((np.cumsum(predicted[:idx, 0])[-1], np.cumsum(predicted[:idx, 1])[-1]))
    v_act = np.hstack((np.cumsum(actual[:idx, 0])[-1], np.cumsum(actual[:idx, 1])[-1]))
    return sqrt(mean_squared_error(v_pred, v_act))


if __name__ == "__main__":

    #plot_results(['log', 'log_static_grid', 'log_dynamic_grid'], ['vanilla', 'static_grid', 'dynamic_grid'])
    #plot_results(['log', 'log_attention_applied_before_lstm'], ['vanilla', 'attention'])

    folder_name = 'log_attention_applied_before_lstm_static_grid_version2'
    evaluation_data = 'test'
    mode = 'train'
    reset = True
    epochs = 20

    path = "../annotations/hyang/video0/"
    video_path = "../videos/hyang/video0/video.mov"
    loader = Loader(path)
    postprocessor = PostProcessing(loader, N_SAMPLES)

    raw = make_df_from_postprocessor(postprocessor, N_INPUT_FEATURES)
    values = raw.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values[:, :N_INPUT_FEATURES])
    data = series_to_supervised(scaled, N_SAMPLES, N_SAMPLES)

    id_idx_train = int(len(np.unique(postprocessor.id)) * SPLIT)
    id_train = np.unique(postprocessor.id)[:id_idx_train]
    id_test = np.unique(postprocessor.id)[id_idx_train:]
    n_train_frames = np.squeeze(np.where(postprocessor.id == id_train[-1]))[-1]
    train_X, train_y, test_X, test_y = split_to_test_train(data, n_train_frames, N_INPUT_FEATURES)


    if mode == 'train':
        model = train_model()
    else:
        model = evaluate_model()

    if evaluation_data == 'train':
        test_X_in = train_X
        test_y_in = train_y
        n_train_frames = 0
    else:
        test_X_in = test_X
        test_y_in = test_y

    pred_series = np.empty([test_y_in.shape[0], test_y_in.shape[1], test_y_in.shape[2]])
    pred_series_baseline = np.empty([test_y_in.shape[0], test_y_in.shape[1], 2])
    input_series = np.empty([test_X_in.shape[0], test_X_in.shape[1], test_X_in.shape[2]])
    output_series = np.empty([test_y_in.shape[0], test_y_in.shape[1], test_y_in.shape[2]])
    colors = pyplot.cm.gist_ncar(np.linspace(.1, .9, np.max(postprocessor.id+1)))

    id_train = np.unique(postprocessor.id[n_train_frames:])
    id_train = id_train[id_train.astype(int) != 38]
    error_method1_1, error_method1_2, error_method1_3, error_method2_1, error_method2_2, error_method2_3= np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    vidcap = imageio.get_reader(video_path, 'ffmpeg')

    pyplot.figure(figsize=(20, 20))
    tot_error1 = 0
    tot_error2 = 0
    counter=0

    for id in id_train:
        frames = postprocessor.id == id
        raw = make_df_from_postprocessor_within_selection(postprocessor, frames, N_INPUT_FEATURES)
        scaled = scaler.transform(raw.values[:, :N_INPUT_FEATURES])
        data = series_to_supervised(scaled, N_SAMPLES, N_SAMPLES)
        trainX, trainY, _, _ = split_to_test_train(data, data.shape[0], N_INPUT_FEATURES)
        model.reset_states()
        yhat = model.predict(trainX)
        pred_series = np.empty([trainY.shape[0], trainY.shape[1], trainY.shape[2]])
        input_series = np.empty([trainX.shape[0], trainX.shape[1], trainX.shape[2]])
        output_series = np.empty([trainY.shape[0], trainY.shape[1], trainY.shape[2]])
        frames = raw['frame']

        for t in range(0, yhat.shape[0], 10):
            pred_series[t] = scaler.inverse_transform(np.hstack((yhat[t], trainX[t][:, 2:])))[:, 0:2]
            input_series[t] = scaler.inverse_transform(trainX[t])
            output_series[t] = scaler.inverse_transform(np.hstack((trainY[t], trainX[t][:, 2:])))[:, 0:2]
            frame = frames[t + N_SAMPLES]
            image = vidcap.get_data(int(frame))

            idx = np.squeeze(np.where(postprocessor.id == id))
            trajectory = np.vstack((postprocessor.x[idx], postprocessor.y[idx])).T
            x_start = trajectory[t + N_SAMPLES, 0]
            y_start = trajectory[t + N_SAMPLES, 1]

            tx = np.arange(0, N_SAMPLES)
            p3x = np.poly1d(np.polyfit(tx, input_series[t, :, 0], 3))
            dx = p3x(tx)

            p3y = np.poly1d(np.polyfit(tx, input_series[t, :, 1], 3))
            dy = p3y(tx)
            pred_series_baseline[t] = np.vstack((dx, dy)).T


            pyplot.cla()
            pyplot.imshow(image)
            pyplot.plot(trajectory[:, 0], trajectory[:, 1], color='black', label=str(id), linestyle='-', linewidth=1,
                        alpha=0.5, marker='+')
            x_start_input = trajectory[t, 0]
            y_start_input = trajectory[t, 1]
            pyplot.plot(np.cumsum(input_series[t, :, 0]) + x_start_input, np.cumsum(input_series[t, :, 1]) + y_start_input,
                        label='x',
                        color='red', linestyle='-', linewidth=1, marker='+')
            pyplot.plot(np.cumsum(output_series[t, :, 0]) + x_start, np.cumsum(output_series[t, :, 1]) + y_start,
                        label='ground truth',
                        color='green', linestyle='-', linewidth=1, marker='+')
            pyplot.plot(np.cumsum(pred_series_baseline[t, :, 0]) + x_start,
                        np.cumsum(pred_series_baseline[t, :, 1]) + y_start, label='poly3',
                        color='blue', linestyle='--', linewidth=1)
            pyplot.plot(np.cumsum(pred_series[t, :, 0]) + x_start, np.cumsum(pred_series[t, :, 1]) + y_start, label='lstm',
                        color='purple', linestyle='--', linewidth=1)
            pyplot.legend()

            rmse1 = calc_rms(pred_series_baseline[t], output_series[t], 30)
            rmse2 = calc_rms(pred_series[t], output_series[t], 30)

            tot_error1 += rmse1
            tot_error2 += rmse2
            pyplot.xlabel('Test RMSE poly3 = %.3f lstm = %.3f Test total RMSE poly3 = %.3f lstm = %.3f' % (rmse1, rmse2, tot_error1, tot_error2))

            error_method1_1 = np.append(error_method1_1, calc_rms(pred_series_baseline[t], output_series[t], 10))
            error_method1_2 = np.append(error_method1_2, calc_rms(pred_series_baseline[t], output_series[t], 20))
            error_method1_3 = np.append(error_method1_3, calc_rms(pred_series_baseline[t], output_series[t], 30))
            error_method2_1 = np.append(error_method2_1, calc_rms(pred_series[t], output_series[t], 10))
            error_method2_2 = np.append(error_method2_2, calc_rms(pred_series[t], output_series[t], 20))
            error_method2_3 = np.append(error_method2_3, calc_rms(pred_series[t], output_series[t], 30))

            pyplot.ylabel('.3s error poly3 = %.3f lstm = %.3f. .6s error poly3 = %.3f lstm = %.3f. 1s error poly3 = %.3f lstm = %.3f' % (
                error_method1_1[-1], error_method2_1[-1], error_method1_2[-1], error_method2_2[-1], error_method1_3[-1], error_method2_3[-1]))

            if folder_name == 'log_cropped':
                pyplot.xlim(image.shape[1]//2 - 500, image.shape[1]//2 + 500)
                pyplot.ylim(image.shape[0]//2 - 500, image.shape[0]//2 + 500)

            pyplot.draw()
            # pyplot.pause(0.001)
            pyplot.savefig(folder_name + '/t_' + str(counter))
            counter += 1
        if counter > 100:
            break


    print('Test total error poly: %.3f' % tot_error1)
    print('Test total error lstm: %.3f' % tot_error2)
    data = [error_method1_1, error_method2_1, error_method1_2, error_method2_2,error_method1_3, error_method2_3]
    np.save(folder_name + '/data.npy', data)

    pyplot.figure(figsize=(20, 20))
    pyplot.violinplot(data, widths = (1, 1, 1, 1, 1, 1))
    pyplot.xticks([1, 2, 3, 4, 5, 6], ['poly .4s', 'lstm .4s', 'poly .7s', 'lstm .7s','poly 1s', 'lstm 1s'])
    pyplot.draw()
    pyplot.savefig(folder_name+'/violin.png')

    pyplot.figure(figsize=(20, 20))
    pyplot.boxplot(data, widths = (1, 1, 1, 1, 1, 1))
    pyplot.xticks([1, 2, 3, 4, 5, 6], ['poly .4s', 'lstm .4s', 'poly .7s', 'lstm .7s','poly 1s', 'lstm 1s'])
    pyplot.draw()
    pyplot.savefig(folder_name+'/boxplot.png')

