import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SHAPE_INPUT = (256, 256, 3)
IMAGE_SHAPE_OUTPUT = (252, 252, 3)
NUMBER_OF_CLASSES = 2

class Sample(object):
    def __init__(self, id):
        self.id = id
        self.input = np.array([IMAGE_SHAPE_INPUT[0], IMAGE_SHAPE_INPUT[1], IMAGE_SHAPE_INPUT[2]])
        self.target = np.array([IMAGE_SHAPE_OUTPUT[0], IMAGE_SHAPE_OUTPUT[1], IMAGE_SHAPE_OUTPUT[2]])
        self.classes = np.array([IMAGE_SHAPE_OUTPUT[0], IMAGE_SHAPE_OUTPUT[1], NUMBER_OF_CLASSES])


class Helper(object):
    def __init__(self, data_path, reload=False):
        self.data_path = data_path
        self.annotation_path = data_path + 'Annotations/'
        description = 'ImageSets/Segmentation/train.txt'
        self.description_file = pd.read_csv(data_path + description, delim_whitespace=True)
        self.description_file.columns = ["name"]
        self.img_dict = {}
        if reload:
            self.create_data()
        else:
            self.load_data()

    def create_data(self):
        all_IDs = sorted(list(self.description_file['name'].keys()))
        self.img_dict = {key: Sample(key) for key in all_IDs}
        for i in all_IDs:
            frame = self.description_file['name'][i]
            self.img_dict[i].input = self.centeredCrop(plt.imread(self.data_path + 'JPEGImages/' + frame + '.jpg'), IMAGE_SHAPE_INPUT)
            self.img_dict[i].target = self.centeredCrop(plt.imread(self.data_path + 'SegmentationObject/' + frame + '.png'), IMAGE_SHAPE_OUTPUT)

        np.save(self.data_path + 'img_dict.npy', self.img_dict)

    def centeredCrop(self, img, image_size):
        (new_height, new_width, _) = image_size
        width = np.size(img, 1)
        height = np.size(img, 0)

        if width < new_width or height < new_height:
            return np.zeros((new_height, new_width, 3))

        left = int(np.ceil((width - new_width) / 2.))
        top = int(np.ceil((height - new_height) / 2.))
        right = int(np.ceil((width + new_width) / 2.))
        bottom = int(np.ceil((height + new_height) / 2.))
        cImg = img[top:bottom, left:right, :]
        return cImg

    def load_data(self):
        self.img_dict = np.load(self.data_path + 'img_dict.npy').item()

    def get_image(self, id):
        if id > len(self.img_dict) or id < 0:
            return False

        return self.img_dict[id]


from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv2DTranspose
from keras.optimizers import SGD
import numpy as np
from keras.models import Model

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute

def VGG_16(weights_path=None):
    # kernel = 3
    # filter_size = 64
    # pad = 1
    # pool_size = 2
    #
    # model = Sequential()
    # model.add(Layer(input_shape=(3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])))
    #
    # print('ENCODER')
    # # encoder
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(filter_size, (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
    # print(model.output_shape)
    #
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(128, (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
    # print(model.output_shape)
    #
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(256, (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
    # print(model.output_shape)
    #
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(512,  (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # print(model.output_shape)
    #
    # print('DECODER')
    # # decoder
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(512,  (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # print(model.output_shape)
    #
    # #model.add(UpSampling2D(size=(pool_size, pool_size)))
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # #model.add(Conv2D(256,  (kernel, kernel), padding="valid"))
    # model.add(Conv2DTranspose(256,  (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # print(model.output_shape)
    #
    # model.add(UpSampling2D(size=(pool_size, pool_size)))
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(128,  (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # print(model.output_shape)
    #
    # model.add(UpSampling2D(size=(pool_size, pool_size)))
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(filter_size,  (kernel, kernel), padding="valid"))
    # model.add(BatchNormalization())
    # print(model.output_shape)
    #
    #
    # if weights_path:
    #     model.load_weights(weights_path)
    input_img = Input(shape=(IMAGE_SHAPE_INPUT[0], IMAGE_SHAPE_INPUT[1], IMAGE_SHAPE_INPUT[2]))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # x = Conv2D(8, (3, 3), activation='relu')(encoded)
    # #x = UpSampling2D((2, 2))(x)
    # x = Conv2D(8, (3, 3), activation='relu')(x)
    # #x = UpSampling2D((2, 2))(x)
    # x = Conv2D(16, (3, 3), activation='relu')(x)
    # #x = UpSampling2D((2, 2))(x)
    # decoded = Conv2D(3, (3, 3), activation='sigmoid')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.output_shape)
    return autoencoder

from keras.models import model_from_json
from keras.callbacks import TensorBoard

folder_name = 'cnn/data/VOC2012/'

def load_model():
    # load json and create model
    json_file = open('cnn/log/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn/log/model.h5")
    print("Loaded model from disk")
    return loaded_model

def train_model(X_train, Y_train,  X_test, Y_test):
    # Test pretrained model
    weights_path = 'cnn/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = VGG_16()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mae')
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=10,shuffle=True, validation_data=(X_test, Y_test), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn/log/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn/log/model.h5")
    print("Saved model to disk")

    return model

def create_test_train(helper, start, stop):
    X_train, Y_train = [], []
    for i in range(start, stop):
        img = helper.get_image(i)
        im_in = img.input.astype('float64')
        im_out = img.target.astype('float64') * 255.

        # im_in[:,:,0] -= 103.939
        # im_in[:,:,1] -= 116.779
        # im_in[:,:,2] -= 123.68

        # plt.subplot(1, 2, 1)
        # plt.imshow(im_in.astype(int))
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(im_out.astype(int))
        #
        # plt.show()

        # im_out = np.expand_dims(im_out, axis=2)
        X_train.append(im_in)
        Y_train.append(im_in[:im_out.shape[0], :im_out.shape[1], :])

        #Y_train.append(helper.centeredCrop(im_in, (200, 200, 3)))

    X_train = np.asarray(X_train) / 255
    Y_train = np.asarray(Y_train) / 255
    return X_train, Y_train

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2bin(gray):
    gray[gray >= 0.5] = 1
    gray[gray < 0.5] = 0
    return gray

if __name__ == "__main__":
    helper = Helper(folder_name, True)
    n_obs = len(helper.img_dict)
    n_obs_train = int(n_obs * 0.8)
    X_train, Y_train = create_test_train(helper, 0, n_obs_train)
    X_test, Y_test = create_test_train(helper,  n_obs_train, n_obs)

    #model = train_model(X_train, Y_train, X_test, Y_test)
    model = load_model()

    # folder_path_input = 'AB/32/input_static/images.npy'
    # X_test = np.load(folder_path_input)/ 255
    # Y_test = np.load(folder_path_input)

    out = model.predict(X_test )

    for i in range(X_test.shape[0]):
        img = out[i]
        im_in = X_test[i]
        im_out = Y_test[i]

        plt.subplot(1, 3, 1)
        plt.imshow((im_in))

        plt.subplot(1, 3, 2)
        plt.imshow((im_out))
        #plt.imshow((im_out[:, :, 0] * 255).astype(int))

        plt.subplot(1, 3, 3)
        #img_out = rgb2gray(img)

        plt.imshow((img))

        #plt.imshow((img[:, :, 0] * 255).astype(int))

        plt.draw()
        plt.pause(0.001)

