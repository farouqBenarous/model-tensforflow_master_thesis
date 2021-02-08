import cv2 as cv2
import requests
import json
import keras
from keras.callbacks import TensorBoard  # for part 3.5 on TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import base64
import numpy as np


def trainV1(X, Y, imageShape):
    model = Sequential()

    model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                     input_shape=(imageShape['rows'], imageShape['columns'], imageShape['channels'])))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensor_board = TensorBoard('model/alexnet/tensorboard/')

    model.fit(X, Y, batch_size=8, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
              callbacks=[tensor_board])

    model.save('model/alexnet/saved_model/alexNetV1.h5', include_optimizer='true')


def trainV2(X, Y, imageShape, model_name):
    if model_name is None or model_name is "":
        print("building new model")
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                         input_shape=(imageShape['rows'], imageShape['columns'], imageShape['channels'])))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(4096, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(16, activation='softmax'))
    else:
        print("loading the saved model")
        model = keras.models.load_model('model/alexnet/saved_model/' + model_name)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensor_board = TensorBoard('model/alexnet/tensorboard/alexNetV2_2')

    model.fit(X, Y, batch_size=32, epochs=100, verbose=1, validation_split=0.1, shuffle=True,
              callbacks=[tensor_board])

    model.save('model/alexnet/saved_model/' + model_name, include_optimizer='true')
    print("model saved")


def run_inference(image, label, model_version):
    try:
        model = keras.models.load_model('model/alexnet/saved_model/' + model_version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()
        # model.predict_classes(image)
        features_map = model.predict(np.expand_dims(image, axis=0))
        print(features_map)


def run_inference_served(image_array, model_url):
    # prepare headers for http request
    headers = {'content-type': 'application/json'}
    response = requests.post(model_url, data=json.dumps({"instances": np.expand_dims(image_array, axis=0).tolist()}),
                             headers=headers)
    # decode response
    print(json.loads(response.text))


def test(X, Y, model_version):
    try:
        model = keras.models.load_model('model/alexnet/saved_model/' + model_version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()
        scores = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return scores
