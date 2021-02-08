import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime


def get_mean_RGB_dataset(X, Y):  # todo : fix this by reading the images directy
    totals = [
        {
            "r_total1": [], "g_total1": [], "b_total1": [],
        },
        {
            "r_total2": [], "g_total2": [], "b_total2": [],
        },
        {
            "r_total3": [], "g_total3": [], "b_total3": [],
        },
        {
            "r_total4": [], "g_total4": [], "b_total4": [],
        },
        {
            "r_total5": [], "g_total5": [], "b_total5": [],
        },
    ]
    for x in X:
        for y in Y:
            if y == 0:
                totals[0]["r_total1"].append(x[0])
                totals[0]["g_total1"].append(x[1])
                totals[0]["b_total1"].append(x[2])
            if y == 1:
                totals[1]["r_total2"].append(x[0])
                totals[1]["g_total2"].append(x[1])
                totals[1]["b_total2"].append(x[2])
            if y == 2:
                totals[2]["r_total3"].append(x[0])
                totals[2]["g_total3"].append(x[1])
                totals[2]["b_total3"].append(x[2])
            if y == 3:
                totals[3]["r_total4"].append(x[0])
                totals[3]["g_total4"].append(x[1])
                totals[3]["b_total4"].append(x[2])
            if y == 4:
                totals[4]["r_total5"].append(x[0])
                totals[4]["g_total5"].append(x[1])
                totals[4]["b_total5"].append(x[2])

    print(totals)
    for total in totals:
        index = "r_total" + str(totals.index(total) + 1)
        print(" R  Mean Of class ", totals.index(total), np.mean(total[index]))
        print(" G  Mean Of class ", totals.index(total), np.mean(total[index]))
        print(" B  Mean Of class ", totals.index(total), np.mean(total[index]))


def get_mean_RGB(img):
    return np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])


def read_image(image, nrows, ncolumns):
    # we can add/change other types of images to improve the classification
    try:
        # variables
        threshold1 = 23
        threshold2 = 23
        drawn_contours = []
        area_contours = []
        contour_used = []

        # read image and convert it to all the needed format
        show = cv2.imread(image, cv2.IMREAD_COLOR)

        imgBlur = cv2.GaussianBlur(show, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        # get all the contours in the image
        contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # use only the biggest contour/object
        for cnt in contours:
            area_contours.append(cv2.contourArea(cnt))
        contour_used = contours[np.argmax(area_contours)]

        # drawn_image = cv2.drawContours(show, cnt, -1, (255, 0, 255), 1)
        # drawn_contours.append(drawn_image)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area >= max(area_contours):
        #         contour_used = cnt
        #         # drawn_image = cv2.drawContours(show, cnt, -1, (255, 0, 255), 1)
        #         # drawn_contours.append(drawn_image)

    except (BaseException) as error:
        print("Error resizing the images ", error, 'image : ', image)
        cv2.imread(image, cv2.IMREAD_COLOR)
        cv2.waitKey(0)
    else:
        # corp the image using the Biggest contour areea
        x, y, w, h = cv2.boundingRect(contour_used)
        resized_image = cv2.resize(show[y:y + h, x:x + w], (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
        return resized_image
    # print all the images + contours
    # stacked_images = stack_images(0.8, [show, imgBlur, imgGray, imgCanny, imgDil,cropped] + drawn_contours)
    # cv2.imshow("stack images ", stacked_images)
    # cv2.waitKey(0)


def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[2000] == imgArray[0][0].shape[2000]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[2000] == imgArray[0].shape[2000]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def load_almondes_dataset(path):
    class1, class2, class3, class4, class5, class6, class7, class8, class9, class10 \
        , class11, class12, class13, class14, class15, class16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    files = os.listdir(path)
    for dirs in files:
        if dirs == "1-tardy nonpareil":
            class1 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'tardy' in i]
        if dirs == '2-fra gulio grade':
            class2 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'fra' in i]
        if dirs == '3-atocha':
            class3 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'atocha' in i]
        if dirs == '4-picantil':
            class4 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'picantil' in i]
        if dirs == '5-pri morsky':
            class5 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'pri' in i]
        if dirs == '6-peerless':
            class6 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'peerless' in i]
        if dirs == '7-moncaio':
            class7 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'moncaio' in i]
        if dirs == '8-marta':
            class8 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'marta' in i]
        if dirs == '9-ferralise':
            class9 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'ferralise' in i]
        if dirs == '10-garrigaes':
            class10 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'garrigaes' in i]
        if dirs == '11-masbovera':
            class11 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'masbovera' in i]
        if dirs == '12-ai':
            class12 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'ai' in i]
        if dirs == '13-planeta':
            class13 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'planeta' in i]
        if dirs == '14-texas':
            class14 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'texas' in i]
        if dirs == '15-philys':
            class15 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'philys' in i]
        if dirs == '17-desmayo lagueto':
            class16 = [path + '/' + dirs + '/{}'.format(i) for i in os.listdir(path + '/' + dirs) if 'desmayo' in i]

    trainImages = class1[:1] + class2[:1] + class3[:1] + class4[:1] + class5[:1] + class6[:1] + \
                  class7[:1] + class8[:1] + class9[:1] + class10[:1] + class11[:1] + class12[:1] + \
                  class13[:1] + class14[:1] + class15[:1] + class16[:1]

    return trainImages


def read_and_process(listImages, nrows, ncolumns):
    X = []  # images
    Y = []  # labels
    for image in listImages:
        img = read_image(image, nrows, ncolumns)
        # r, g, b = get_mean_RGB(read_image(image, nrows, ncolumns))
        X.append(img)
        # get the labels
        if 'tardy' in image:
            Y.append(0)
        elif 'fra' in image:
            Y.append(1)
        elif 'atocha' in image:
            Y.append(2)
        elif 'picantil' in image:
            Y.append(3)
        elif 'pri' in image:
            Y.append(4)
        elif 'peerless' in image:
            Y.append(5)
        elif 'moncaio' in image:
            Y.append(6)
        elif 'marta' in image:
            Y.append(7)
        elif 'ferralise' in image:
            Y.append(8)
        elif 'garrigaes' in image:
            Y.append(9)
        elif 'masbovera' in image:
            Y.append(10)
        elif 'ai' in image:
            Y.append(11)
        elif 'planeta' in image:
            Y.append(12)
        elif 'texas' in image:
            Y.append(13)
        elif 'philys' in image:
            Y.append(14)
        elif 'desmayo' in image:
            Y.append(15)
    X = np.array(X)
    Y = np.array(to_categorical(Y, 16))
    return X, Y


def get_average_size_of_images(listImages):
    widths, heights = [], []
    for image in listImages:
        shape = read_image(image, 224, 224).shape
        print(shape)
        heights.append(shape[0])
        widths.append(shape[1])
    return np.min(heights), np.min(widths)


def plot_pictures(images):
    plt.figure(figsize=(20, 10))
    columns = 5
    for i in range(columns):
        plt.subplot(5 / columns + 1, columns, i + 1)
        plt.imshow(images[i])
        plt.show()


def rename_pictures():
    path = 'data/Images/'
    names = os.listdir(path)
    i = 1
    for indexName, fileName in enumerate(names):
        images = os.listdir(path + '/' + fileName)
        print(images)
        for index, img in enumerate(images):
            os.rename(os.path.join(path + fileName + '/', img),
                      os.path.join(path + fileName + '/', fileName.join([str(index), '.jpg'])))


def save_array_file(array):
    np.savetxt('image_numpyed.txt', array, fmt='%d')