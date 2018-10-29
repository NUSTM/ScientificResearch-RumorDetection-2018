__author__ = "zhangc0321@gmail.com"

import tensorflow as tf
import gensim
import numpy as np
import linecache  
import random
import csv

train_dir = './data/'

labels = tf.placeholder(tf.float32)
dh = 40
d = 50
"""
t0 = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32, [None])
"""
f = tf.placeholder(tf.float32, [50, None])
f0 = tf.placeholder(tf.float32, [50, None])
"""
#time attention
c = tf.Variable(tf.zeros([100]), name='c') #100->max time interval
ci = t - t0"""
#content attention
b = tf.Variable(0.,name='b')
Wh = tf.Variable(tf.zeros([dh,2*d]),name='Wh')
Wa = tf.Variable(tf.zeros([dh]),name='Wa')
W = tf.Variable(tf.zeros([d]),name='W')
Hei = tf.tanh(tf.matmul(Wh, tf.concat([f, f0], 0)))
Ac = tf.reduce_sum(tf.multiply(Wa, tf.transpose(Hei)), reduction_indices=1)
Vei = tf.nn.softmax(Ac)
Rei = tf.reduce_sum(tf.multiply(f, Vei), reduction_indices=1)
temp = tf.reduce_sum(tf.multiply(W, Rei)) + b
Lei = tf.sigmoid(temp)

first_train_vars = tf.get_collection(Wh, Wa)
second_train_vars = tf.get_collection(W, b)
reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.001), tf.trainable_variables())
crossloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=temp, labels=labels))
loss = crossloss+reg
tf.summary.scalar('loss',loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.009
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.9, staircase=True)
opt = tf.train.GradientDescentOptimizer(learning_rate)
filter_vars1 = ['Wa', 'Wh']
filter_vars2 = ['W', 'b']
train_vars1 = []
train_vars2 = []
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    for filter_var in filter_vars1:
        if filter_var in var.name:
            train_vars1.append(var)
    for filter_var in filter_vars2:
        if filter_var in var.name:
            train_vars2.append(var)
            
opt_op1 = opt.minimize(loss, var_list=train_vars1, global_step=global_step, name="opt_op1")
opt_op2 = opt.minimize(loss, var_list=train_vars2, global_step=global_step, name="opt_op2")

init_op = tf.global_variables_initializer()

def compute_accuracy():
    i = 0.
    correct = 0
    with open('WeiboTest.txt','r',encoding='utf-8') as file:
        while True:
            linestr = file.readline()
            if not linestr:
                break
            linelist = linestr.split()
            eid = int(linelist[0][4:])
            label = int(linelist[1][-1:])
            ff = np.loadtxt(open(f'./{label}/{eid}.csv',"rb"),delimiter=",",skiprows=0)
            label = float(label)
            newdata = np.vsplit(ff, 2)
            f_data = newdata[1]
            f0_data = newdata[0]
            yi = sess.run(Lei, feed_dict = {labels : label, f : f_data, f0 : f0_data})
            yi = round(yi)
            i += 1
            if(yi == int(label)):
                correct += 1
    return correct/i
                
def preprocessing():
    count = -1
    for count,line in enumerate(open(r'Weibo.txt','r')):
        pass
    count += 1
    fo1 = open('WeiboTrain.txt','w',encoding='utf-8')
    fo0 = open('WeiboTest.txt','w',encoding='utf-8')
    with open('Weibo.txt','r',encoding='utf-8') as f:
        for i in range(count):
            linestr = f.readline()
            a = random.randrange(5)
            if a == 3:
                fo0.write(linestr)
            else:
                fo1.write(linestr)
    fo0.close()
    fo1.close()
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    preprocessing()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("Desktop/",sess.graph) 
    #sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint("model/"))
    traincount = -1
    for traincount,line in enumerate(open(r'WeiboTrain.txt','r')):
        pass
    traincount += 1
    i = 0
    print(traincount)
    for n in range(1000000):
        i += 1
        a = random.randrange(1, traincount)
        theline = linecache.getline(r'WeiboTrain.txt', a)
        linelist = theline.split()
        eid = int(linelist[0][4:])
        label = int(linelist[1][-1:])
        ff = np.loadtxt(open(f'./{label}/{eid}.csv',"rb"),delimiter=",",skiprows=0)
        label = float(label)
        newdata = np.vsplit(ff, 2)
        f_data = newdata[1]
        f0_data = newdata[0]
        for epoch in range(1):
            avg_cost = 0.
            sess.run(opt_op1, feed_dict = {labels : label, f : f_data, f0 : f0_data})
            sess.run(opt_op2, feed_dict = {labels : label, f : f_data, f0 : f0_data})
            if i % 10 == 0:  
                result = sess.run(merged,feed_dict={labels : label, f : f_data, f0 : f0_data})
                writer.add_summary(result,i)
            if i % 1000 == 0:
                print(compute_accuracy())
            if i % 50000 == 0:
                saver=tf.train.Saver()
                saver.save(sess, "model/my-model", global_step=i)
                print("save the model")
                    