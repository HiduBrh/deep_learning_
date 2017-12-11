
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.losses import *
from keras.optimizers import *
from keras.metrics import *
from keras.callbacks import *

import pickle

x_train = pickle.load(open('x_train.txt', 'rb'))
x_test = pickle.load(open('x_test.txt', 'rb'))
y_train = pickle.load(open('y_train.txt', 'rb'))
y_test = pickle.load(open('y_test.txt', 'rb'))

tb_callback = TensorBoard("C:/Project/kerasTest/Logs/cats_dogs_mlp_1hl_2units_bias/")

x_train = x_train.reshape(x_train.shape[0], 3072)
x_test = x_test.reshape(x_test.shape[0], 3072)

model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(3072,), use_bias=True))
#model.add(Dropout(0.2))
#model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation=sigmoid, use_bias=True))
#model.add(Dropout(0.2))

model.compile(sgd(), loss=mse, metrics=['accuracy'])
model.fit(x_train, y_train, 250, 4000, 1, [tb_callback], 0., (x_test, y_test), True, None, None, 0)
