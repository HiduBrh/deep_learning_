import os, cv2
import numpy as np
import random
import math
import pickle

random.seed=302
DARK=[0,0,0]

TRAIN_DIR = 'C:/Users/idu/Documents/ESGI/M2/deep learning/dgs vs cts/train/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

indexes=[]
for i in range(len(train_images)):
    indexes.append(i)
random.shuffle(indexes)
x_train=[]
y_train=[]

for i in indexes[:math.floor(len(indexes)*0.7)]:
    x_train.append(train_images[i])
    y_train.append([labels[i]])

x_test=[]
y_test=[]

for i in indexes[math.floor(len(indexes)*0.7):]:
    x_test.append(train_images[i])
    y_test.append([labels[i]])

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img.shape[0] > img.shape[1]:
        img2 = cv2.copyMakeBorder(img, 0, 0, 0, img.shape[0] - img.shape[1], cv2.BORDER_CONSTANT, value=DARK)
    else:
        img2 = cv2.copyMakeBorder(img, 0, img.shape[1] - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=DARK)

    return cv2.resize(img2, (32, 32), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, 32, 32, 3), dtype=np.uint8)
    data=np.array(data)
    for i, image_file in enumerate(images):
        data[i] = read_image(image_file)
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))

    return data

x_train=prep_data(x_train)
x_test=prep_data(x_test)
x_train=x_train.astype('float32')
x_train=x_train/255.0
x_test=x_test.astype('float32')
x_test=x_test/255.0

with open('x_train_padding.txt','wb') as f:
    pickle.dump(x_train,f)
with open('x_test_padding.txt','wb') as f1:
    pickle.dump(x_test,f1)
with open('y_train.txt','wb') as f2:
    pickle.dump(y_train,f2)
with open('y_test.txt','wb') as f3:
    pickle.dump(y_test,f3)

print (x_train)

