# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:31:59 2021

@author: DELL
"""

import tensorflow as tf
import keras
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data() 





from keras import models
from keras import layers
#Netword Layers and Models#

network=models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


#The compilation step#
from keras import optimizers
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#Prepping the data according to the network requirements#
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


#Preparing labels#
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Test the network#
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)



###############################################################################


'''import numpy as np
x = np.array(12) #0 dimension
x.ndim #Gives the dimension of the array

x=np.array([1,2,3]) #1 dimension
x=np.array([[1,2,3],[4,5,6]]) # 2 dimensional (Matrix)
x = np.array([[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]]]) #Note: number of axes are called rank
'''

'''my_slice=train_images[:, :, :] # [:,:,:] the first colon is for the list,
#the second is for the coloumn of the matrix, the third one is the row.
#my_slice = train_images[:, 7:-7, 7:-7] #The pixels centered at the middle.
print(my_slice.shape)'''
'''
digit = train_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()'''