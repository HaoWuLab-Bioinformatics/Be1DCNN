from bi1dcnn import OneDCNN
import numpy as np
import random
import keras
from keras.utils import to_categorical

def train(X,F,chr_name):
    #X: 正样本
    #F: 负样本
    seed = 7
    np.random.seed(seed)
    x = np.vstack((X[0], F[0]))
    test_x = np.vstack((X[1], F[1]))
    y = np.array([1]*X[0].shape[0] + [0]*F[0].shape[0])#标签
    test_y = np.array([1]*X[1].shape[0] + [0]*F[1].shape[0])
    tmp=y
    tmp.reshape(y.shape[0],1)
    label = keras.utils.to_categorical(tmp, num_classes=2)
    tmp=test_y
    tmp.reshape(test_y.shape[0],1)
    test_label = keras.utils.to_categorical(tmp, num_classes=2)
    index = [i for i in range(len(x))]  
    random.shuffle(index) 
    x = x[index]
    label = label[index]
    print('load over')
    learning_rate=0.001
    epochs=100
    kernel_size = 5
    '''
    train model
    '''
    features = 243
    train_x = np.zeros((len(x), features, 1))
    train_x[:,:,0]=x[:, :]
    test_x_r = np.zeros((len(test_x), features, 1))
    test_x_r[:, :, 0] = test_x[:, :]
    train_y=label
    test_y=test_label
    print('train_x_shape:',train_x.shape)
    print('test_x_r_shape:',test_x_r.shape)
    model=OneDCNN.One_DCNN_model(learning_rate,epochs,train_x,train_y,test_x_r, test_y,chr_name,kernel_size)
    model.train_model()
