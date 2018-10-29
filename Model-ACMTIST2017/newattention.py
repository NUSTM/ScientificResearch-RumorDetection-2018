train_dir = './'

import tensorflow as tf
import gensim
import numpy as np
import linecache  
import random
import os
import sys
def get_files(file_dir):
    neg = []
    label_neg = []
    pos = []
    label_pos = []
    for file in os.listdir(file_dir+'/0'):
            neg.append(file_dir +'/0'+'/'+ file) 
            label_neg.append(0)
    for file in os.listdir(file_dir+'/1'):
            pos.append(file_dir +'/1'+'/'+file)
            label_pos.append(1)
    #把cat和dog合起来组成一个list（img和lab）
    data_list = np.hstack((pos, neg))
    label_list = np.hstack((label_pos, label_neg))

    #利用shuffle打乱顺序
    temp = np.array([data_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    #从打乱的temp中再取出list（img和lab）
    data_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    return data_list, label_list

def get_batch(data, label, batch_size, capacity):
    #转换类型
    data = tf.cast(data, tf.string)
    label = tf.cast(label, tf.int32)
 
    # make an input queue
    input_queue = tf.train.slice_input_producer([data, label])
 
    label = input_queue[1]
    data_contents = tf.read_file(input_queue[0]) #read img from a queue  
    data = tf.decode_csv(data_contents, record_defaults=[ [ ]*50 ]) 
    print(data)
    data_batch, label_batch = tf.train.batch([data, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    data_batch = tf.cast(data_batch, tf.float32)
    return data_batch, label_batch            

data_list, label_list = get_files(train_dir)
ff, label = get_batch(data_list, label_list, 1, 512)
print(ff)
dh = 40
d = 50
f, f0 = tf.split(ff, 2, 0)

#content attention
b = tf.Variable(0.,name='b')
Wh = tf.Variable(tf.zeros([dh,2*d]),name='Wh')
Wa = tf.Variable(tf.zeros([dh]),name='Wa')
W = tf.Variable(tf.zeros([d]),name='W')
Hei = tf.tanh(tf.matmul(Wh, tf.concat([f, f0], 0)))
Ac = tf.reduce_sum(tf.multiply(Wa, tf.transpose(Hei)), reduction_indices=1)
Vei = tf.nn.softmax(Ac)
Rei = tf.reduce_sum(tf.multiply(Vei, f), reduction_indices=1)
temp = tf.reduce_sum(tf.multiply(W, Rei)) + b
Lei = tf.sigmoid(temp)

first_train_vars = tf.get_collection(Wh, Wa)
second_train_vars = tf.get_collection(W, b)
reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.5), tf.trainable_variables())
crossloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Lei, labels=labels))
loss = crossloss
tf.summary.scalar('loss',loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1.
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.9, staircase=True)
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
    i = 0
    correct = 0
    with open('WeiboTest.txt','r',encoding='utf-8') as file:
        while True:
            linestr = file.readline()
            if not linestr:
                break
            linelist = linestr.split()
            eid = int(linelist[0][4:])
            label = int(linelist[1][-1:])
            with open(f'./data/{label}/{eid}.txt','r',encoding='utf-8') as ff:
                line = ff.readline()
                f0_data = model.infer_vector(line)
                f00 = f0_data
                f_data = f0_data
                while True:
                    line = ff.readline()
                    if not line:
                        break
                    fi = model.infer_vector(line)
                    f0_data = np.vstack((f0_data, f00))
                    f_data = np.vstack((f_data, fi))
                label = float(label)
                f_data = np.transpose(f_data)
                f0_data = np.transpose(f0_data)
                yi = sess.run(Lei, feed_dict = {labels : label, f : f_data, f0 : f0_data})
                yi = round(yi)
                i += 1
                if(yi == int(label)):
                    correct += 1
    return correct/60.
                
 
with tf.Session() as sess:
    print("Load doc2vec Model...")
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("Desktop/",sess.graph) 
    model = gensim.models.doc2vec.Doc2Vec.load('./my_db.d2v')
    sess.run(init_op)
    i = 0;
    for i in range(100000):
        a = random.randrange(1, 4665)
        theline = linecache.getline(r'WeiboTrain.txt', a)
        linelist = theline.split()
        eid = int(linelist[0][4:])
        label = int(linelist[1][-1:])
        with open(f'./data/{label}/{eid}.txt','r',encoding='utf-8') as ff:
            line = ff.readline()
            f0_data = model.infer_vector(line)
            f00 = f0_data
            f_data = f0_data
            while True:
                line = ff.readline()
                if not line:
                    break
                fi = model.infer_vector(line)
                f0_data = np.vstack((f0_data, f00))
                f_data = np.vstack((f_data, fi))
            label = float(label)
            f_data = np.transpose(f_data)
            f0_data = np.transpose(f0_data)
            for epoch in range(1):
                i += 1
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
    