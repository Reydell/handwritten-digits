# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import torch

import transforms as tfs

from itertools import product

import torch.nn as nn
import torch.nn.functional as func
import torchviz

WINDOW_SIZE = 400  # 392

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 16, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(1, 28, 28)
        x = self.pool(func.leaky_relu(self.conv1(x)))
        x = self.pool(func.leaky_relu(self.conv2(x)))
        #         print(x.shape)
        x = x.view(-1, 4 * 4 * 16)
        x = func.leaky_relu(self.fc1(x))
        x = func.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNadv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(1, 28, 28)
        #         print(0, x.shape)
        x = self.pool(func.leaky_relu(self.conv1(x)))
        #         print(1, x.shape)
        x = self.pool(func.leaky_relu(self.conv2(x)))
        #         print(2, x.shape)
        x = func.leaky_relu(self.conv3(x))
        #         print(3, x.shape)
        x = x.view(-1, 8 * 3 * 3)
        #         print('last: ', x.shape)
        x = func.leaky_relu(self.fc1(x))
        x = func.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

def softmax(arr):
    res = torch.exp(arr)
    return res / res.sum()

def to_nn(nparr):
    return torch.from_numpy(nparr).view([1, 1, 28, 28]).type_as(torch.FloatTensor())

def predict(model):
    with open('img.pickle', 'rb') as fin:
        res = pickle.load(fin)

    # plt.figure(figsize=(7, 7))
    # sns.heatmap(res, cmap=sns.color_palette("Greys", as_cmap=True))
    # plt.show()

    input_img = to_nn(res)

    acc = 0
    x_batch = input_img
    y_batch = torch.FloatTensor([7])

    x_batch = x_batch.view(x_batch.shape[0], -1).to(torch.device('cpu'))
    y_batch = y_batch.to('cpu')

    preds = torch.argmax(model(x_batch), dim=1)
    acc += (preds==y_batch).numpy().mean()

    print(f'Prediction: {preds[0]}')
    return preds[0]

def predict_prob(model):
    with open('img.pickle', 'rb') as fin:
        res = pickle.load(fin)

    # plt.figure(figsize=(7, 7))
    # sns.heatmap(res, cmap=sns.color_palette("Greys", as_cmap=True))
    # plt.show()

    input_img = to_nn(res)

    x_batch = input_img
    y_batch = torch.FloatTensor([7])

    x_batch = x_batch.view(x_batch.shape[0], -1).to(torch.device('cpu'))
    y_batch = y_batch.to('cpu')

    preds = torch.argmax(model(x_batch), dim=1)

    prob = softmax(model.forward(x_batch)[0])
    print(prob.shape)


    print(f'Prediction: {preds[0], prob[preds[0]]}')

    # torchviz.make_dot(model(x_batch), params=dict(model.named_parameters())).render("model", format="png")

    return preds[0], prob[preds[0]]

def blur(img, w=1):
    weights = np.array([[.2, .55, .2], [.55, .0, .55], [.2, .55, .2]])
    s = weights.sum()
    weights /= s
    res = img.copy()
    for i, j in product(range(w, len(img) - w), repeat=2):
        if res[i, j] > .95:
            continue

        res[i, j] = np.sum(img[i-1: i+2, j-1: j+2] * weights)

    return res

def make_darker(img, w=1):
    weights = np.array([[.2, .55, .2], [.55, 1., .55], [.2, .55, .2]])
    s = weights.sum()
    weights /= s
    res = img.copy()
    for i, j in product(range(w, len(img) - w), repeat=2):
        if res[i, j] < .005 or res[i, j] > .95:
            continue

        res[i, j] = np.sum(img[i - 1: i + 2, j - 1: j + 2] * weights)

    return res

def apply_blur(img):
    return make_darker(make_darker(blur(img)))

def to_torch(arr, label):
    return torch.from_numpy(np.interp(arr, [0, 1], [-1, 1])).view((1, 28, 28)).to(torch.float32), label

# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.res = np.zeros((28, 28), dtype=float)
        self.digit = np.zeros((28, 28), dtype=float)

        # self.nn = torch.load('model3_s', map_location=torch.device('cpu'))

        self.nn1 = torch.load('cnn_3_convs_ss')


        self.mnist_train = []
        self.mnist_test = []

        self.is_loaded = False

        self.testchange = 0
        self.trainchange = 0

        self.predicted_label = 0

        # setting title
        self.setWindowTitle("Paint with PyQt5")

        # setting geometry to main window
        self.setGeometry(100, 100, WINDOW_SIZE, WINDOW_SIZE)

        # creating image object
        self.image = QImage(WINDOW_SIZE, WINDOW_SIZE, QImage.Format_RGB32)

        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 30
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()

        # creating menu bar
        mainMenu = self.menuBar()

        # creating file menu for save and clear action
        fileMenu = mainMenu.addMenu("File")

        digitMenu = mainMenu.addMenu("Digit")

        # digit buttons
        act0 = QAction(f'{0}', self)
        act0.setShortcut(f'{0}')
        digitMenu.addAction(act0)
        act0.triggered.connect(lambda: self.learn(0))

        act1 = QAction(f'{1}', self)
        act1.setShortcut(f'{1}')
        digitMenu.addAction(act1)
        act1.triggered.connect(lambda: self.learn(1))

        act2 = QAction(f'{2}', self)
        act2.setShortcut(f'{2}')
        digitMenu.addAction(act2)
        act2.triggered.connect(lambda: self.learn(2))

        act3 = QAction(f'{3}', self)
        act3.setShortcut(f'{3}')
        digitMenu.addAction(act3)
        act3.triggered.connect(lambda: self.learn(3))

        act4 = QAction(f'{4}', self)
        act4.setShortcut(f'{4}')
        digitMenu.addAction(act4)
        act4.triggered.connect(lambda: self.learn(4))

        act5 = QAction(f'{5}', self)
        act5.setShortcut(f'{5}')
        digitMenu.addAction(act5)
        act5.triggered.connect(lambda: self.learn(5))

        act6 = QAction(f'{6}', self)
        act6.setShortcut(f'{6}')
        digitMenu.addAction(act6)
        act6.triggered.connect(lambda: self.learn(6))

        act7 = QAction(f'{7}', self)
        act7.setShortcut(f'{7}')
        digitMenu.addAction(act7)
        act7.triggered.connect(lambda: self.learn(7))

        act8 = QAction(f'{8}', self)
        act8.setShortcut(f'{8}')
        digitMenu.addAction(act8)
        act8.triggered.connect(lambda: self.learn(8))

        act9 = QAction(f'{9}', self)
        act9.setShortcut(f'{9}')
        digitMenu.addAction(act9)
        act9.triggered.connect(lambda: self.learn(9))

        saveAction = QAction("Save", self)
        saveAction.setShortcut("S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        loadAction = QAction("Load", self)
        fileMenu.addAction(loadAction)
        loadAction.triggered.connect(self.load_data)

        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Left")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        plotAction = QAction("Plot", self)
        plotAction.setShortcut("P")
        fileMenu.addAction(plotAction)
        plotAction.triggered.connect(self.plot)

    def load_data(self):
        if self.is_loaded:
            print('Data already loaded')
            return

        with open('mtrain.pickle', 'rb') as fin:
            self.mnist_train = pickle.load(fin)

        with open('mtest.pickle', 'rb') as fin:
            self.mnist_test = pickle.load(fin)

        print(f'Test length: {len(self.mnist_test)}')
        print(f'Train length: {len(self.mnist_train)}')

        self.is_loaded = True

    def learn(self, d):
        # temp = to_torch(self.digit, d)
        r = np.random.randint(1, 7) # for now not from 0 to 7
        for pic in tfs.d123u123r123l123(self.digit):
            r = np.random.randint(0, 7)
            if r:
                self.trainchange += 1
                self.mnist_train.append(to_torch(pic, d))
            else:
                self.testchange += 1
                self.mnist_test.append(to_torch(pic, d))

        self.setWindowTitle(f'{d} added to {"train" if r else "test"}')

        print(f'added new picture of {d}')

        self.clear()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:

            y, x = event.pos().x(), event.pos().y()
            # print(x, y)

            x = np.interp(x, [0, WINDOW_SIZE], [0, 27.99])
            y = np.interp(y, [0, WINDOW_SIZE], [0, 27.99])
            # print(x, y)

            xi, xf = divmod(x, 1)
            yi, yf = divmod(y, 1)
            xi = int(xi)
            yi = int(yi)
            # print('\t', xi, xf)

            # dist = abs(np.interp(xf, [0, 1], [-.5, .5])) + abs(np.interp(yf, [0, 1], [-.5, .5]))

            # color = 1 - dist
            color = 1.

            self.res[xi, yi] = color

            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())

            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

            self.pred_prob()

    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def save(self):
        if not self.is_loaded:
            self.load_data()

        self.setWindowTitle('SAVING...')
        if self.testchange:
            np.random.shuffle(self.mnist_test)
            with open('mtest.pickle', 'wb') as fout:
                pickle.dump(self.mnist_test, fout)

        if self.trainchange:
            np.random.shuffle(self.mnist_train)
            with open('mtrain.pickle', 'wb') as fout:
                pickle.dump(self.mnist_train, fout)


        self.setWindowTitle(f'SAVED')
        print(f'added {self.testchange} to test and {self.trainchange} to train')
        self.testchange = 0
        self.trainchange = 0

    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

        self.res = np.zeros_like(self.res)  ######### ######### ######### ######### ######### #########

    def plot(self):

        self.digit = apply_blur(self.res)

        # with open('img.pickle', 'wb') as p:
        #     pickle.dump(np.interp(self.digit, [0, 1], [-1, 1]), p)

        plt.figure(figsize=(10, 10))
        sns.heatmap(self.digit, cmap=sns.color_palette('Greys', as_cmap=True))
        plt.show()

        # self.setWindowTitle(f'Prediction: {predict(self.nn1)}')

    def pred(self):
        self.digit = apply_blur(self.res)

        with open('img.pickle', 'wb') as p:
            pickle.dump(np.interp(self.digit, [0, 1], [-1, 1]), p)

        # prob = self.nn1.forward()
        res = predict(self.nn1)
        self.predicted_label = res[0]

        self.setWindowTitle(f'Prediction: {predict(self.nn1)}')
        print(f'predicted_label: {self.predicted_label}')

    def pred_prob(self):
        self.digit = apply_blur(self.res)

        with open('img.pickle', 'wb') as p:
            pickle.dump(np.interp(self.digit, [0, 1], [-1, 1]), p)

        # prob = self.nn1.forward()
        res = predict_prob(self.nn1)
        self.predicted_label = res[0]

        self.setWindowTitle(f'Prediction: {res[0]},  {res[1]:.0%} sure')
        print(f'predicted_label: {self.predicted_label}')

App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())
