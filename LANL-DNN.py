# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:54:53 2019

@author: user
"""

from os import walk
import sys
import os
import numpy as np
import pandas as pd
from tensorflow.contrib.layers import fully_connected
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

##將path設為目前資料夾
PATH="D:\data"



X = tf.placeholder(tf.float32, [None, 150000])
Y = tf.placeholder(tf.float32, [None, 1])
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

w_h = init_weights([150000, 1000])
w_h2 = init_weights([1000, 500])
w_o = init_weights([500, 1])


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    
    return tf.matmul(h2, w_o)

saver = tf.train.Saver()
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_sum(tf.square(py_x -  Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
epoch_count = 20000
batch_size = 50
keep_input = 0.8
keep_hidden = 0.75

def set_data(data):
    x=data[:,0]
    y=data[:,1]
    x=x.reshape(1,-1)
    y=np.mean(y)
    return x,y

print(0)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
step = 0
print(1)
chunkSize = 150000 #一次讀資料的筆數
loss=1000
bestCost=sys.maxsize
f = open(PATH+'/train.csv')
reader = pd.read_csv(f, iterator=True)
loop=True
while loop:
    step+=1
    for i in range(batch_size):
        try:
            #print(3)
            #每次只讀指定的比數後下去train
            chunk = reader.get_chunk(chunkSize)
            if chunk.values.shape[0] < chunkSize:
                break
            train_x, train_y = set_data(chunk.values)
            if i == 0:
                batch_x=train_x
                batch_y=train_y
            else:
                batch_x=np.row_stack((batch_x,train_x))
                batch_y=np.row_stack((batch_y,train_y))
            #print(batch_x)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
            break
    #print(batch_y)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, p_keep_input: keep_input, p_keep_hidden: keep_hidden})
    loss = sess.run([cost], feed_dict={X: batch_x, Y: batch_y, p_keep_input: 1., p_keep_hidden: 1.})
    if loss[0]<bestCost:
        bestCost=loss[0]
        saver.save(sess, PATH+"/tmp/model.ckpt")
        lastImprove=step
    if step-lastImprove>10:
        print("early stop")
        break
    print("Epoch:",step, "\tLoss:",loss[0])
f.close()


"""寫資料"""
saver.restore(sess, PATH+"/tmp/model.ckpt")

data=[]
line=['seg_id','time_to_failure']
data.append(line)
mypath="D:\data/test"
for filename in os.listdir(mypath):
    loadFile = open(os.path.join(mypath, filename), 'r')
    reader = pd.read_csv(loadFile)
    test=reader.values
    test=test.reshape(1,-1)
    predict=sess.run([py_x], feed_dict={X: test, p_keep_input: 1., p_keep_hidden: 1.})
    loadFile.close()
    line=[filename.split('.',1)[0],predict[0][0][0]]
    data.append(line)
data_df = pd.DataFrame(data)
data_df.to_csv('D:\data/submission.csv',index=False,header=False)
#sess.close()