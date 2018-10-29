#-*- coding: utf-8 -*-
# This is a version implement a paper:"Detecting Rumors from Microblogs with Recurrent Neural Networks"(https://ijcai.org/Proceedings/16/Papers/537.pdf)

import json,os
from os import listdir
import os
import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
# For add layer sequentally
from keras.models import Sequential
# For add fully connected layer
from keras.layers import Dense,Activation,SimpleRNN

from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam,Adagrad
import tensorflow as tf
import thulac

tf.app.flags.DEFINE_integer('N', 12, "TIME_STEPS")
FLAGS = tf.app.flags.FLAGS

env_dist = os.environ
CX_WORD_DIR = env_dist['CX_WORD_DIR']

Weibo_Json_Dir = "./data/Weibo"
Weibo_TXT = "./data/Weibo.txt"

# For some unknown reason, some case can't be handled normally
# BlackList = ["3489762868350457", "3495366957480235", "3524005681870859", "3564599817357079", "3613146499346398", "3650475917913991", "3921825278712911", "3600441415772315", "3603614419737740", "3685964956754161", "3651171602494589", "3664010711257804", "3486845608638980", "3486986402905181", "3488336150630619", "3489163980623509", "3492335394659283", "3491543916953598", "3491623524636311", "3491643027771174", "3493463742206046", "3495381470293153", "3496628361121803", "3499245883634226", "3456869509414752", "3667475285315591"]

BlackList = []

def GetEventList():
	EventList = []
	weibo_txt = open(Weibo_TXT, "r")
	event_count = 0
	for line in weibo_txt.readlines():
		event_count += 1
		event_str = line.split()
		eid = (event_str[0].split(':'))[1]
		if eid in BlackList:
			continue
		label = (event_str[1].split(':'))[1]
		posts = event_str[2:]
		Event = {"eid" : eid, "label" : label, "posts" : posts} #Event结构
		EventList.append(Event) 
	weibo_txt.close()
	return EventList

def ContinuousInterval(intervalL):
	maxInt = []
	tempInt = [intervalL[0]]
	for q in range(1,len(intervalL)):
		if intervalL[q]-intervalL[q-1] > 1:
			if len(tempInt) > len(maxInt):
				maxInt = tempInt
			tempInt = [intervalL[q]]
		else:
			tempInt.append(intervalL[q])
	if len(maxInt)==0:
		maxInt = tempInt
	return maxInt

def main(_):
	np.random.seed(3)	#固定seed让每次的random都一样
	TIME_STEPS = FLAGS.N
	IMPUT_SIZE = 625	
	BATCH_SIZE = 30
	BATCH_INDEX = 0
	OUTPUT_SIZE = 2
	CELL_SIZE = 175
	LR = 0.001

	totalData = []
	totalDataLabel = []
	counter = 0
	totalDoc = 0
	totalpost = 0
	tdlist1 = 0
	Pos = 0
	Neg = 0
	maxpost = 0
	minpost = 62827

	thulac_pip = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
	EventList = GetEventList()

	print("Generating BlackList with N = ", TIME_STEPS, " ...")
	for event in EventList:
		totalDoc += 1
		Eid = event["eid"]
		Label = event["label"]
		# print("Eid : ", Eid, "Label: ", Label)
		WeiboPostIdList = event["posts"]
		if len(WeiboPostIdList) == 1:
			tdlist1 += 1
			continue
		if len(WeiboPostIdList) >= maxpost:
			maxpost = len(WeiboPostIdList)
		if len(WeiboPostIdList) <= minpost:
			minpost = len(WeiboPostIdList)

		event_file_path = os.path.join(Weibo_Json_Dir, Eid + ".json")
		event_file = open(event_file_path, "r")
		event_json = json.load(event_file)

		WeiboPostList = []
		index = 0
		for WeiboPostId in WeiboPostIdList:
			totalpost += 1
			WeiboJson = event_json[index]
			index += 1
			WeiboText = WeiboJson["text"]
			Time = WeiboJson["t"]
			WeiboPost = {"text" : WeiboText, "time" : Time}
			WeiboPostList.append(WeiboPost)
		if Label == "0":
			Pos += 1
		else:
			Neg += 1
		#Sort by time
		WeiboPostList = sorted(WeiboPostList, key=lambda k: k['time'])

		#find Time Invertal of weibo
		TotalTimeLine = WeiboPostList[-1]['time']-WeiboPostList[0]['time']
		IntervalTime = TotalTimeLine/TIME_STEPS
		k = 0
		PreConInt = []
		while True:
			k += 1
			WeiboIndex = 0
			output = []
			if TotalTimeLine == 0:	
				for weibo in WeiboPostList:
					weibo_text = thulac_pip.cut(weibo["text"], text=True)
					output.append(weibo_text)
				break
			Start = WeiboPostList[0]['time']
			Interval = int(TotalTimeLine/IntervalTime)
			Intset = []
			for inter in range(0,Interval):
				empty = 0
				interval = []
				for q in range(WeiboIndex,len(WeiboPostList)):
					if WeiboPostList[q]['time'] >= Start and WeiboPostList[q]['time'] < Start+IntervalTime:
						empty += 1
						weibo_text = thulac_pip.cut(WeiboPostList[q]["text"], text=True)
						interval.append(weibo_text)
					#记录超出interval的weibo位置，下次可直接从此开始
					elif WeiboPostList[q]['time'] >= Start+IntervalTime:
						WeiboIndex = q-1
						break
				# empty interval
				if empty == 0:
					output.append([])
				else:
					#add the last weibo
					if WeiboPostList[-1]['time'] == Start+IntervalTime:
						weibo_text = thulac_pip.cut(WeiboPostList[-1]["text"], text=True)
						interval.append(weibo_text)
					Intset.append(inter)
					output.append(interval)
				Start = Start+IntervalTime
			ConInt = ContinuousInterval(Intset)
			if len(ConInt)<TIME_STEPS and len(ConInt) > len(PreConInt):
				IntervalTime = int(IntervalTime*0.5)
				PreConInt = ConInt
				if IntervalTime == 0:
					output = output[ConInt[0]:ConInt[-1]+1]
					break
			else:
				# print(len(ConInt))
				output = output[ConInt[0]:ConInt[-1]+1]
				break
		counter+=1
		event_file.close()
		# print (counter)
		# 把Interval的所有字都串在一起
		for q in range(0,len(output)):
			output[q] = ''.join(s for s in output[q])

		try:
		#Caculate Tfidf
			vectorizer = CountVectorizer()
			transformer = TfidfTransformer()
		#print(output)
			tf = vectorizer.fit_transform(output)
			tfidf = transformer.fit_transform(tf)
		# Debug
		# print(tfidf.toarray())
			Allvocabulary = vectorizer.get_feature_names()
		except ValueError:
			BlackList.append(Eid)
			continue

		# print(vectorizer.get_feature_names())
		Input = []

		for interval in tfidf.toarray():
			interval = sorted(interval,reverse=True)
			while len(interval) < IMPUT_SIZE:
				interval.append(0.0)
			Input.append(interval[:IMPUT_SIZE])
		if len(Input) < TIME_STEPS:
			for q in range(0,TIME_STEPS-len(Input)):
				Input.insert(0,[0.0] * IMPUT_SIZE)
		totalData.append(Input[:TIME_STEPS])
		totalDataLabel.append(Label)

	file_name = CX_WORD_DIR + "/BlackLists/BlackList" + str(FLAGS.N) + ".txt"
	f = open(file_name,'w')
	f.write(str(BlackList))
	f.close()
	print("Generating BlackList with N = ", TIME_STEPS, " done.")


if __name__ == '__main__':
	tf.app.run()