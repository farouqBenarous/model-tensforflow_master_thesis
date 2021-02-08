import keras
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard  # for part 3.5 on TensorBoard
import tensorflow as tf
import numpy as np


# designed to train on  the image as it is
def trainV1(X, Y, imageShape, model_name):
    if model_name is None or model_name is "":
        model = Sequential()
        model.add(InputLayer(input_shape=(imageShape['rows'], imageShape['columns'], imageShape['channels'])))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', ))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu', ))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))

        model.add(Dense(16, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        print("loading the saved model")
        model = keras.models.load_model('model/fullyConnectedLayers/fullyConnectedSavedModel/' + model_name)

    model.summary()
    tensor_board = TensorBoard('model/fullyConnectedLayers/tensorboardFullyConnectedModel')

    model.fit(X, Y, batch_size=32, epochs=1, verbose=1, validation_split=0.1, shuffle=True,
              callbacks=[tensor_board])

    model.save('model/fullyConnectedLayers/fullyConnectedSavedModel/'+"annV1.h5", include_optimizer='true')
    print("model saved")

# designed to train ON the mean RGB of the image
def trainV2(X, Y, imageShape):
    model = Sequential()
    model.add(InputLayer(input_shape=(3,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensor_board = TensorBoard('model/fullyConnectedLayers/tensorboardFullyConnectedModel')

    model.fit(X, Y, batch_size=8, epochs=10000, verbose=1, validation_split=0.2, shuffle=True,
              callbacks=[tensor_board])

    model.save('model/fullyConnectedLayers/fullyConnectedSavedModel/annV2.h5', include_optimizer='true')


def run_inference(image, version):
    try:
        model = keras.models.load_model('model/fullyConnectedLayers/fullyConnectedSavedModel/' + version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()

        model.predict_classes(image)


def test(X, Y, version):
    try:
        model = keras.models.load_model('model/fullyConnectedLayers/fullyConnectedSavedModel/' + version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()
        scores = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return scores
