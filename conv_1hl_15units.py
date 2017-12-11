
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras.metrics import *
from keras.callbacks import *

import pickle

x_train = pickle.load(open('x_train.txt','rb'))
x_test = pickle.load(open('x_test.txt','rb'))
y_train = pickle.load(open('y_train.txt','rb'))
y_test = pickle.load(open('y_test.txt','rb'))

tb_callback = TensorBoard("C:/Project/kerasTest/Logs/cats_dogs_convolution_lhl_15units/")

model = Sequential()
model.add(Conv2D(16, (3, 3), activation=tanh, input_shape=(32, 32, 3)))
model.add(MaxPool2D((2, 2)))
#model.add(AveragePooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(15, activation=tanh))
model.add(Dense(1, activation=sigmoid))
model.compile(sgd(), loss=mse, metrics=['accuracy'])

model.fit(x_train, y_train, 250, 4000, 1, [tb_callback], 0., (x_test, y_test), True, None, None, 0)

