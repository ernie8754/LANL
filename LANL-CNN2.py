# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:50:00 2019

@author: user
"""

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Conv1D,MaxPooling1D,GlobalAveragePooling1D,Dropout, LSTM
import sys
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
#from keras.optimizers import adam
#from keras import losses



'''def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X,0.01))
    strain.append(np.quantile(X,0.05))
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    return pd.Series(strain)

#Training data for DNN
train = pd.read_csv('train.csv', iterator=True, chunksize=150000, dtype={'acoustic_data': np.int16})
num_statistical_features = 13
X_train_features = pd.DataFrame()
Y_train_features = pd.Series()
for df in train:
    features = gen_features(df['acoustic_data'])
    X_train_features = X_train_features.append(features, ignore_index=True)
    Y_train_features = Y_train_features.append(pd.Series(df['time_to_failure'].values[-1]))
#Scale input data
scaler = StandardScaler()
scaler.fit(X_train_features)
X_train_scaled = scaler.transform(X_train_features)
X_train_scaled = np.float32(X_train_scaled.reshape(-1,num_statistical_features))
Y_train = np.float32(Y_train_features.values) 
n_samples = X_train_scaled.shape[0]
print(X_train_scaled.shape,Y_train.shape)
print(X_train_scaled.dtype,Y_train.dtype)'''



earthquake_CNN_model = Sequential()
earthquake_CNN_model.add(Conv1D(100, 10, activation='relu', input_shape=(150000, 1)))
earthquake_CNN_model.add(MaxPooling1D(100))
earthquake_CNN_model.add(Conv1D(160, 10, activation='relu'))
earthquake_CNN_model.add(GlobalAveragePooling1D())
earthquake_CNN_model.add(Dense(16,kernel_initializer='normal', activation='relu'))
#earthquake_CNN_model.add(Dropout(0.5))
earthquake_CNN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

earthquake_CNN_model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mse'])
print(earthquake_CNN_model.summary())


def set_data(data):
    x=data[:,0].reshape(1,-1,1)
    y=data[:,1]
    y=np.array(np.mean(y)).reshape(1,1)
    #y=round(y,0)
    #y=np.array(to_categorical(y, num_classes=17)).reshape(1,-1)
    return x,y


step = 0
print(1)
chunkSize = 150000 #一次讀資料的筆數
batch_size=5
bestCost=sys.maxsize
while True:
    step+=1
    count=0
    loss_sum=0
    loop=True
    f = open(os.path.join('train.csv'))
    reader = pd.read_csv(f, iterator=True)
    while loop:
        count+=1
        try:
            chunk = reader.get_chunk(chunkSize)
            if chunk.values.shape[0] < chunkSize:
                break
            train_x, train_y = set_data(chunk.values)
            #print(chunk.index)
            loss = earthquake_CNN_model.train_on_batch(x=train_x,y= train_y)
            loss_sum+=loss[0]
            pred=earthquake_CNN_model.predict(train_x)
            sys.stdout.write("\rEpoch:{2} Total Count:{0} Loss:{1} pred:{3} true:{4}".format(count,loss_sum/count,step,pred,train_y))
            sys.stdout.flush()
            #print(batch_x)
        except StopIteration:
            print("Iteration is stopped.")
            f.close()
            loop=False
            break
    sys.stdout.write("\rEpoch:{2} Total Count:{0} Loss:{1}\n".format(count,loss_sum/count,step))
    if loss_sum/count<bestCost:
        bestCost=loss_sum/count
        lastImprove=step
        earthquake_CNN_model.save('CNN_model.h5')
    if step-lastImprove>=10:
        print("early stop")
        break
#history = earthquake_CNN_model.fit(X_train_scaled,Y_train, epochs=500,verbose=1) 
    #print("Epoch:",step, "\tLoss:",loss_sum)
earthquake_CNN_model = load_model(os.path.join('CNN_model.h5'))
data=[]
line=['seg_id','time_to_failure']
data.append(line)
for filename in os.listdir(os.path.join('test')):
    sys.stdout.write("\rFile Name:{0}".format(filename))
    loadFile = open(os.path.join('test', filename), 'r')
    reader = pd.read_csv(loadFile)
    test=reader.values
    test=test.reshape(1,-1,1)
    #test_1=np.arange(len(test)).reshape(-1,1)
    #test=np.append(test,test_1,axis=1).reshape(1,-1,2)
    predict=earthquake_CNN_model.predict(test)
    loadFile.close()
    line=[filename.split('.',1)[0],predict[0][0]]
    data.append(line)
data_df = pd.DataFrame(data)
data_df.to_csv('D:\data/submission.csv',index=False,header=False)

'''
true=pd.read_csv('D:\data/submission.csv')
data=np.array(data,dtype=float)[:,1]
true=np.array(true,dtype=float)[:,1]
true=true.reshape(-1)
data=data[1:]
k=true-data
abs(k)
k_sum=abs(k)
k_sum=np.sum(k_sum)
k_sum/2624
'''