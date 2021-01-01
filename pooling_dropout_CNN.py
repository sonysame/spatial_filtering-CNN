import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def show_prediction():
	n_show=96
	y=model.predict(x_test)
	plt.figure(1,figsize=(12,8))
	plt.gray()
	for i in range(n_show):
		plt.subplot(8,12,i+1)
		x=x_test[i,:]
		x=x.reshape(28,28)
		plt.pcolor(1-x)
		wk=y[i,:]
		prediction=np.argmax(wk)
		plt.text(22,25.5, "%d"%prediction, fontsize=12)
		if prediction!=np.argmax(y_test[i,:]):
			plt.plot([0,27],[1,1], color='cornflowerblue', linewidth=5)
		plt.xlim(0,27)
		plt.ylim(27,0)
		plt.xticks([], "")
		plt.yticks([], "")

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


checkpoint_dir = './training_checkpoints2'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model=Sequential()
model.add(Conv2D(16,(3,3),padding='same',input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
print(model.summary())

startTime=time.time()
history=model.fit(x_train,y_train, batch_size=1000, epochs=20, verbose=1, validation_data=(x_test, y_test))
model.save_weights(checkpoint_prefix)
score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Computation time:{0:.3f}.'.format(time.time()-startTime))

show_prediction()
plt.show()