from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from sklearn.utils import resample
import numpy as np
from keras.layers import  Flatten, Dense, Dropout, Conv1D

BATCH_SIZE = 50

class One_DCNN_model():
    def __init__(self, learning_rate, epochs, train_x, train_y, test_x, test_y, chromname,kernel_size):
        self.rate = learning_rate
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.chromname = chromname
        self.kernel_size=kernel_size
    def evaluateModel(self,trainX,trainY):
        input_shape = (243, 1)
        model = Sequential()
        model.add(Conv1D(256, kernel_size=self.kernel_size, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Conv1D(256, kernel_size=self.kernel_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv1D(256, kernel_size=self.kernel_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.rate), metrics=['accuracy'])
        model.summary()
        model.fit(trainX, trainY, epochs=self.epochs,batch_size=64)
        return model
    def ensemblePredict(self,models):
        yhats=[model.predict(self.test_x)for model in models]
        yhats= np.array(yhats)
        sum=np.sum(yhats,axis=0)
        result=np.argmax(sum,axis=1)
        return result
    def train_model(self):
        n_split=10
        scores=[]
        for m in range(n_split):
            ix = [i for i in range(len(self.train_x))]
            train_ix = resample(ix, replace=True)
            trainX, trainY = self.train_x[train_ix], self.train_y[train_ix]
            model=self.evaluateModel(trainX, trainY)
            save_path = r'../model_1dcnn_6/' + self.chromname + '_'+str(m)+'.h5'
            model.save(save_path)