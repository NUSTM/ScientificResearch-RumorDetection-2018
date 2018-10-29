__author__ = "zhangc0321@gmail.com"

import os, sys, shutil, collections
import jieba, traceback
import json

counter_dict = collections.defaultdict(int)
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

src_dir = {'Weibo':'./Weibo'}
fo1 = open('1.txt','w',encoding='utf-8')
fo0 = open('0.txt','w',encoding='utf-8')

with open('Weibo.txt','r',encoding='utf-8') as f:
    while True:
        linestr = f.readline()
        if not linestr:
            break
        linelist = linestr.split()
        eid = int(linelist[0][4:])
        label = int(linelist[1][-1:])
        with open('./Weibo/%d.json'%(eid),'r',encoding='utf-8') as ff:
            data = json.load(ff)
            for i in range(len(data)):
                text = data[i]["text"]
                
                filter_chars = "\r\n\t，。；！,.:;：、“”‘’"
                trans_dict = dict.fromkeys((ord(_) for _ in filter_chars), '')
                text = text.translate(trans_dict)
                
                it = jieba.cut(text, cut_all=False)
                _ = []
                for w in it:
                    _.append(w)
                
                if label == 1:
                    fo1.write(' '.join(_) + '\n')
                else:
                    fo0.write(' '.join(_) + '\n')
fo0.close()
fo1.close()