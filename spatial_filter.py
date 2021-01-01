import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
"""
train_data_file=open("../input/mnist_train.csv",'r')
train_list=train_data_file.readlines()
train_data_file.close()

test_data_file=open("../input/mnist_test.csv",'r')
test_list=test_data_file.readlines()
test_data_file.close()

x_train=np.zeros((60000,784))
y_train=np.zeros((60000))

x_test=np.zeros((10000,784))
y_test=np.zeros((10000))

for i in range(len(train_list)):
	x_train[i]=train_list[i].strip().split(",")[1:]
	y_train[i]=train_list[i].strip().split(",")[0]
	
for i in range(len(test_list)):
	x_test[i]=test_list[i].strip().split(",")[1:]
	y_test[i]=test_list[i].strip().split(",")[0]

np.savez('mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
"""
outfile=np.load('mnist_data.npz')

x_train=outfile['x_train']
y_train=outfile['y_train']
x_test=outfile['x_test']
y_test=outfile['y_test']

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
num_classes=10

y_train=tf.keras.utils.to_categorical(y_train, num_classes)
y_test=tf.keras.utils.to_categorical(y_test, num_classes)

id_img=2
myfil1=np.array([[1,1,1],[1,1,1],[-2,-2,-2]],dtype=float)
myfil2=np.array([[-2,1,1],[-2,1,1],[-2,1,1]], dtype=float)

x_img=x_train[id_img, :, :, 0] #28x28

img_h=28
img_w=28
out_img1=np.zeros_like(x_img)
out_img2=np.zeros_like(x_img)

for ih in range(img_h-3):
	for iw in range(img_w-3):
		img_part=x_img[ih:ih+3, iw:iw+3]
		out_img1[ih+1, iw+1]=np.dot(img_part.reshape(-1), myfil1.reshape(-1))
		out_img2[ih+1, iw+1]=np.dot(img_part.reshape(-1), myfil2.reshape(-1))


plt.figure(1,figsize=(12,3.2))
plt.subplots_adjust(wspace=0.5)
plt.gray()

plt.subplot(1,3,1)

plt.pcolor(-x_img)
plt.xlim(-1,29)
plt.ylim(29,-1)

plt.subplot(1,3,2)
plt.pcolor(-out_img1)
plt.xlim(-1,29)
plt.ylim(29,-1)

plt.subplot(1,3,3)
plt.pcolor(-out_img2)
plt.xlim(-1,29)
plt.ylim(29,-1)

plt.show()
