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

id_img=2
myfil1=np.array([[1,1,1],[1,1,1],[-2,-2,-2]],dtype=float)
myfil2=np.array([[-2,1,1],[-2,1,1],[-2,1,1]], dtype=float)

x_img=x_train[id_img, :, :, 0] #28x28

img_h=28
img_w=28
out_img1=np.zeros_like(x_img)
out_img2=np.zeros_like(x_img)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model=Sequential()
model.add(Conv2D(8,(3,3),padding='same', input_shape=(28,28,1),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
print(model.summary())
startTime=time.time()
print(np.shape(x_train))
#history=model.fit(x_train,y_train, batch_size=1000, epochs=20, verbose=1, validation_data=(x_test, y_test))
#model.save_weights(checkpoint_prefix)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Computation time:{0:.3f} sec'.format(time.time()-startTime))

show_prediction()
#plt.show()

plt.figure(2, figsize=(12,2.5))
plt.gray()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.subplot(2,9,10)
id_img=12
x_img=x_test[id_img, :, :, 0] #28x28
img_h=28
img_w=28
plt.pcolor(-x_img)
plt.xlim(0,img_h)
plt.ylim(img_w, 0)
plt.xticks([], "")
plt.yticks([], "")
plt.title("Original")

w=model.layers[0].get_weights()[0]

max_w=np.max(w)
min_w=np.min(w)

for i in range(8):
	plt.subplot(2,9,i+2)
	w1=w[:,:,0,i]
	w1=w1.reshape(3,3)
	plt.pcolor(-w1, vmin=min_w, vmax=max_w)
	plt.xlim(0,3)
	plt.ylim(3,0)
	plt.xticks([],"")
	plt.yticks([],"")
	plt.title("%d"%i)
	plt.subplot(2,9,i+11)
	out_img=np.zeros_like(x_img)

	for ih in range(img_h-3):
		for iw in range(img_w-3):
			img_part=x_img[ih:ih+3, iw:iw+3]
			out_img[ih+1, iw+1]=np.dot(img_part.reshape(-1), w1.reshape(-1))

	plt.pcolor(-out_img)
	plt.xlim(0,img_w)
	plt.ylim(img_h,0)
	plt.xticks([],"")
	plt.yticks([], "")
plt.show()


