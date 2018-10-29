__author__ = "zhangc0321@gmail.com"

import tensorflow as tf
import gensim
import numpy as np
import linecache  
import random
model = gensim.models.doc2vec.Doc2Vec.load('./my_db.d2v')
with open(r'Weibo.txt', 'r', encoding='utf-8') as f:
    while True:
        theline = f.readline()
        if not theline:
            break
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
            f_data = np.transpose(f_data)
            f0_data = np.transpose(f0_data)
            np.savetxt(f'./{label}/{eid}.csv', np.row_stack((f0_data, f_data)),delimiter = ',')