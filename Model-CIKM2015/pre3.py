
import os, sys, shutil, collections
import jieba, traceback
import json
import gensim

# compile sample documents into a list
doc_set = []
tem_set = []
with open('G:/jiayou/Weibo.txt','r',encoding='utf-8') as f:
    while True:
        linestr = f.readline()#一个Event
        if not linestr:
            break
        linelist = linestr.split()#分隔符对字符串进行切片
        eid = int(linelist[0][4:])#取第1个元素的第5位字符及以后 
        with open('G:/jiayou/pre2/%d.json'%(eid),'r',encoding='utf-8') as ff:
            data = json.load(ff)
            for i in range(len(data)):
                text = data[i]["text_list"]
                textlist = text.split()
                doc_set.append(textlist)

from gensim import corpora, models
 # Create a corpus from a list of texts
dictionary = corpora.Dictionary(doc_set)
corpus = [dictionary.doc2bow(text) for text in doc_set]
#print(corpus)

 # Train the model on the corpus.
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=18) 

#print(ldamodel.print_topics(num_topics=2))
with open('G:/jiayou/Weibo.txt','r',encoding='utf-8') as f:
    while True:
        linestr = f.readline()#一个Event
        if not linestr:
            break
        linelist = linestr.split()#分隔符对字符串进行切片
        eid = int(linelist[0][4:])#取第1个元素的第5位字符及以后 
        with open('G:/jiayou/pre2/%d.json'%(eid),'r',encoding='utf-8') as ff:
            data = json.load(ff)
            fo1 = open('G:/jiayou/pre3/%d.json'%(eid),'w',encoding='utf-8')#一个Event一个.txt
            result = []
            for j in range(len(data)):
                rtem = {}
                test_doc = doc_set[j]
                doc_bow = dictionary.doc2bow(test_doc)
                doc_lda = ldamodel[doc_bow]
                tem = []
                re = []
                for doc in doc_lda:
                    #tem.append(doclda[i][1])
                    rtem['%d'%(doc[0])] = str(doc[1])
                #re.append(tem)
                #rtem["lda"]=re.tolist()
                result.append(rtem)
                #print("Result",rtem)
            fo1.write(json.dumps(result,ensure_ascii=False,indent=4))
            fo1.close()
        ff.close()







