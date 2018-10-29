# coding=utf-8
import os, sys, shutil, collections
import jieba, traceback
import json
import pandas as pd
import numpy as np
from sklearn import svm

x=[]
y=[]
def load_data():
    with open('G:/jiayou/Weibo.txt','r',encoding='utf-8') as f:
        while True:
            linestr = f.readline()#一个Event
            if not linestr:
                break
            linelist = linestr.split()#分隔符对字符串进行切片
            eid = int(linelist[0][4:])#取第1个元素的第5位字符及以后
            label = int(linelist[1][-1:])#取第2个元素倒数第一个字符及以后
            
            with open('G:/jiayou/pre3/%d.json'%(eid),'r',encoding='utf-8') as f1:
                with open('G:/jiayou/pre2/%d.json'%(eid),'r',encoding='utf-8') as f2:
                    data1 = json.load(f1)
                    data2 = json.load(f2)
                    for i in range(len(data1)):
                        tem = []
                        for j in range(0,18):
                            if(("%d"%j) in data1[i]):
                                tem.append(data1[i]["%d"%j])
                            else:
                                tem.append(0)
                        tem.append(data2[i]["length"])
                        tem.append(data2[i]["positive_words"])
                        tem.append(data2[i]["negative_words"])
                        tem.append(data2[i]["sentiment_score"])
                        tem.append(data2[i]["url_sum"])
                        tem.append(data2[i]["firstperson"])
                        tem.append(data2[i]["hashtag"])
                        tem.append(data2[i]["mention"])
                        tem.append(data2[i]["questin_sum"])
                        tem.append(data2[i]["exclamation_sum"])
                        tem.append(data2[i]["multi_question"])
                        tem.append(data2[i]["multi_exclamation"])
                        tem.append(data2[i]["user_description"])
                        tem.append(data2[i]["picture"])
                        tem.append(data2[i]["verified"])
                        x.append(tem)
                        y.append(label)

    from sklearn import cross_validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test

# 使用LinearSVC考察线性分类SVM的预测能力
def test_LinearSVC(X_train,X_test,y_train,y_test):
    # 选择模型
    cls = svm.LinearSVC()
    # 把数据交给模型训练
    clf_linear = svm.SVC(kernel='linear').fit(X_train,y_train)
    test_pred=clf_linear.predict(X_test)
    count = 0
    correct = 0
    for i in range(len(test_pred)):
        count +=1
        if test_pred[i] == y_test[i]:
            correct +=1
    return correct/count
'''
    #clf_rbf = svm.SVC(kernel='rbf').fit(X_train,y_train)
    #clf_sigmoid = svm.SVC(kernel='sigmoid').(X_train,y_train)
    #cls.fit(X_train,y_train)
    
    #print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
    #print('Score: %.2f' cls.score(X_test, y_test))
'''
if __name__=="__main__":
    X_train,X_test,y_train,y_test=load_data() # 生成用于分类的数据集
    print(test_LinearSVC(X_train,X_test,y_train,y_test) )# 调用 test_LinearSVC
