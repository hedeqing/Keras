from keras.optimizers import SGD
from keras import Sequential
from keras.models import Model
from keras.layers import Activation,Dense,Dropout

import keras
import tensorflow as tf
import numpy as np

#创建模型
model=Sequential()

#构建数据，utils.to_categorical能进行热编码处理
x_train=np.random.random((1000,20))
y_train=keras.utils.to_categorical(np.random.randint(2,size=(1000,1)),num_classes=10)
x_test=np.random.random((100,20))
y_test=keras.utils.to_categorical(np.random.randint(2,size=(100,1)),num_classes=10)

#通过add来构建网络模型Dense（uint，activation，input_dim),uint是输出维度，activation是使用的激活函数，input_dim是输入的维度，
model.add(Dense(64,input_dim=20,activation='relu'))
#使用激活函数Dropout
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#使用SGD优化器，momentum参数为
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20,batch_size=128)
score=model.evaluate(x_test,y_test,batch_size=128)

#创建会话
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    test_writer=tf.summary.FileWriter(log_dir+'test_writer',tf.get_default_graph())
    print(sess.run(score))
