# coding: utf-8

# In[19]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import *
from keras.layers.advanced_activations import *
import numpy as np,scipy as sp,pandas as pd
from keras.models import model_from_json
import os
import timeit
import time
def WiseNet(weights_path=None):
    model = Sequential()
    Init='glorot_normal'
    model.add(Convolution2D(32, 6, 1,init=Init,input_shape=(1,145, 9)))
    model.add(PReLU(init=Init))

    print model.output_shape
    model.add(MaxPooling2D((6, 1), strides=(1,1)))
    print model.output_shape

    model.add(ZeroPadding2D((0, 1)))
    model.add(Convolution2D(32, 6, 3,init=Init))
    model.add(PReLU(init=Init))
    print model.output_shape
    model.add(ZeroPadding2D((1, 0)))
    model.add(MaxPooling2D((2, 1), strides=(2, 1)))
    print model.output_shape

    model.add(Flatten())
    model.add(Dense(512,init=Init))
   # model.add(Dropout(0.5))
    model.add(PReLU(init=Init))
    model.add(Dense(512,init=Init))
   # model.add(Dropout(0.5))
    model.add(PReLU(init=Init))
    model.add(Dense(1024,init=Init))
   # model.add(Dropout(0.5))
    model.add(PReLU(init=Init))
    model.add(Dense(1, activation='sigmoid'))
    print model.output_shape

    if weights_path:
        model.load_weights(weights_path)

    # Remove the last two layers to get the 4096D activations
    # model.layers.pop()
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    return model


# In[22]:
if __name__ == "__main__":
    '''
    model = WiseNet()
   # allowed_kwargs = {'clipnorm':5, 'clipvalue':5}
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=5, clipvalue=5)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipnorm=5, clipvalue=5)
    adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08,clipnorm=5, clipvalue=5)
    model.compile(optimizer=adadelta, loss='binary_crossentropy',metrics=['accuracy'])
   '''
    nrow=None
    a = pd.read_csv('./feature.csv', nrows=nrow, header=None)
    a_len=len(a)
    b=np.array(a)

    y=b[:,0:1]
    #y=np.hstack((train_y,1-train_y))
    y=y
    x=b[:,1:]
    x -= np.mean(x, axis=0)  # zero-center
    #normalize
    x_std = np.std(x, axis=0)
    x_std[x_std == 0] = 1
    len(x_std[x_std == 0])
    x /= x_std  # normalize

    x=x.reshape((len(x),9,145))
    x=x.transpose(0,2,1)#转置
    x=np.expand_dims(x,1)#在索引1处扩维


    test_x = x[0:int(a_len * 0.3), :]
    test_y = y[0:int(a_len * 0.3), :]
    train_x = x[int(a_len * 0.3):, :]
    train_y = y[int(a_len * 0.3):, :]
    
    model = WiseNet()
   # allowed_kwargs = {'clipnorm':5, 'clipvalue':5}
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=5, clipvalue=5)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipnorm=5, clipvalue=5)
    adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08,clipnorm=5, clipvalue=5)
    model.compile(optimizer=adadelta, loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(train_x,train_y,batch_size=100, nb_epoch=5)

    json_string = model.to_json()
    open('my_wisenet_architecture_epoch100.json', 'w').write(json_string)
    model.save_weights('my_wisenet_weights_epoch5.h5',overwrite=True)

# In[ ]:
    model = model_from_json(open('my_wisenet_architecture_epoch100.json').read())
    model.load_weights('my_wisenet_weights_epoch5.h5')
    model.compile(optimizer=adadelta, loss='binary_crossentropy', metrics=['accuracy'])
    result=model.predict(test_x)

    result_train=model.predict(train_x)
    bo_train=result_train==train_y
    bo_train=bo_train[bo_train==True]
    print len(bo_train),len(train_x)

    bo = result == test_y
    bo = bo[bo == True]
    print len(bo),len(test_y)
    print float(len(bo)) / len(test_y)
