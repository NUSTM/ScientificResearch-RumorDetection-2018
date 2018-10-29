
import os, sys, shutil, collections
import jieba, traceback
import json
import re
import urllib
import urllib.request

counter_dict = collections.defaultdict(int)
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)


def stopwordslist(filepath):#停用词表
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords

def positivewordslist(filepath):#正面情感词词表
    words = [line.strip() for line in open(filepath, 'r').readlines()]
    return words

def negativewordslist(filepath):#负面情感词词表
    words = [line.strip() for line in open(filepath, 'r').readlines()]
    return words

def geturl(str):#网页正则式匹配
    results=re.findall("(?isu)(http\://[a-zA-Z0-9\.\?/&\=\:]+)",str)
    return results

with open('G:/jiayou/Weibo.txt','r',encoding='utf-8') as f:
    while True:
        linestr = f.readline()#一个Event
        if not linestr:
            break
        linelist = linestr.split()#分隔符对字符串进行切片
        eid = int(linelist[0][4:])#取第1个元素的第5位字符及以后
        label = int(linelist[1][-1:])#取第2个元素倒数第一个字符及以后
        result = []
        with open('G:/jiayou/Weibo/%d.json'%(eid),'r',encoding='utf-8') as ff:
            data = json.load(ff)
            fo1 = open('G:/jiayou/pre1/%d.json'%(eid),'w',encoding='utf-8')#一个Event一个.txt
            
            text=""
            stopwords = stopwordslist('G:/jiayou/stopwords.txt')
            positivelist = positivewordslist('G:/jiayou/正面情感词语（中文）.txt')
            negativelist = negativewordslist('G:/jiayou/负面情感词语（中文）.txt')
            firstpersonlist = ["我","我们","俺","俺们"]
            questinlist = ["?","？"]
            exclamationlist = ["!","！"]
            for i in range(len(data)):
                blog_length = positive_words = negative_words = sentiment_score=0
                url_sum = smiling_sum=frowning_sum=firstperson_num=0
                questin_sum=exclamation_sum=0
                tem = {}
                text = data[i]["text"]
                urls = geturl(text)
                for url in urls:
                    print(url)
                    text=text.strip(url)
                
                tem["length"] = len(text)
                text_list = jieba.cut(text)
                out = ""
                for word in text_list:
                    if word in questinlist:
                        questin_sum += 1
                    if word in exclamationlist:
                        exclamation_sum += 1
                    if word in positivelist:
                        positive_words += 1
                    if word in negativelist:
                        negative_words += 1
                    if word in firstpersonlist:
                        firstperson_num +=1
                    if word not in stopwords:
                        if word !='\t':
                           out += word+" "
                
                tem["text_list"] = out
                tem["positive_words"] = positive_words
                tem["negative_words"] = negative_words
                if len(urls)>0: 
                    tem["url_sum"] = 1
                else:
                    tem["url_sum"] = 0
                tem["firstperson"] = firstperson_num
                if "#" in text:
                    tem["hashtag"] = 1
                else:
                    tem["hashtag"] = 0
                if "@" in text:
                    tem["mention"] = 1
                else:
                    tem["mention"] = 0
                tem["questin_sum"] = questin_sum
                tem["exclamation_sum"] = exclamation_sum
                if questin_sum > 1:
                    tem["multi_question"] = 1
                else:
                    tem["multi_question"] = 0
                if exclamation_sum > 1:
                    tem["multi_exclamation"] = 1
                else:
                    tem["multi_exclamation"] = 0
                user_description = str(data[i]["user_description"])
                if(len(user_description)==0):
                    tem["user_description"] = 0
                else:
                    tem["user_description"] = 1
                if data[i]["picture"]!= None:
                    tem["picture"] = 1
                else:
                    tem["picture"] = 0
                if bool(data[i]["verified"]) == False:
                    tem["verified"] = 0
                else:
                    tem["verified"] = 1
                tem["verified_type"] = data[i]["verified_type"]
                tem["gender"] = data[i]["gender"]
                if int(data[i]["city"]) == 1 :
                    tem["city"] = 1
                else:
                    tem["city"] = 0
                tem["friends_count"] = data[i]["friends_count"]
                tem["followers_count"] = data[i]["followers_count"]
                tem["existtime"] = data[i]["t"] - data[i]["user_created_at"]
                if data[i]["favourites_count"]==0:
                    tem["reputation"] = 0
                else:
                    tem["reputation"] = data[i]["followers_count"]/data[i]["favourites_count"]
                tem["retweets"] = data[i]["reposts_count"]
                tem["comments_count"] = data[i]["comments_count"]
                tem["t"] = data[i]["t"]
                result.append(tem)
                
                
            fo1.write(json.dumps(result,ensure_ascii=False,indent=4))
            fo1.close()
        ff.close()