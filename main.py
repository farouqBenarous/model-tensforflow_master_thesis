import sys

import numpy as np
from sklearn.model_selection import train_test_split

from model.alexnet import AlexnetModel
from model.alexnet.AlexnetModel import run_inference_served
from model.fullyConnectedLayers import FullyConnectedLayersModel
from shared.functions import load_almondes_dataset, read_and_process

if __name__ == '__main__':
    np.random.seed(42)

    # sizing of the images
    nrows = 224
    ncolumns = 224
    channels = 3  # change to 1 if you want to use grayscale image

    print("loading the data ...")
    dataSet = load_almondes_dataset('data/Images')

    print("preprocessing the data ...")
    X, Y = read_and_process(dataSet, nrows, ncolumns)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

    if sys.argv[1] == 'ann':
        print("Fully connected Model is being called ")
        FullyConnectedLayersModel.trainV1(X_train, y_train, {'rows': nrows, 'columns': ncolumns, 'channels': channels},
                                          None)
        FullyConnectedLayersModel.test(X_test, y_test, "annV1.h5")
        quit()
    elif sys.argv[1] == 'alexnet':
        print("Alexnet Model is being called ")
        # AlexnetModel.trainV2(X, Y, {'rows': nrows, 'columns': ncolumns, 'channels': channels}, "alexNetV2_1.h5")
        # AlexnetModel.test(X, Y, "alexNetV2_1.h5")
        # run_inference_served(X_train[0],'http://localhost:8501/v1/models/alexnet:predict')
        # save_array_file(X_train[0])
        AlexnetModel.run_inference(X[0], Y[0], "alexNetV2_1.h5")
        quit()

    print("Please enter a valid model name")
