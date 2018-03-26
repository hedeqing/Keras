import  numpy as np
import keras
import tensorflow as tf

from keras.layers import Activation,Dense,Dropout,Flatten
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD

x_train=np.random.random((1000,100,100,3))
y_train=keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
x_test=np.random.random((200,100,100,3))
y_test=keras.utils.to_categorical(np.random.randint(10,size=(200,1)),num_classes=10)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#the Flatten is useless actually

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
                optimizer='sgd')
model.fit(x_train,y_train,epochs=10,batch_size=32)
score=model.evaluate(x_test,y_test,batch_size=32)

with tf.Session() as sess:
    sess.run(init)
    print("start")
    print(sess.run(score))
    print("finish")
