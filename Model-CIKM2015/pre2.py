
import os, sys, shutil, collections
import jieba, traceback
import json

counter_dict = collections.defaultdict(int)
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

with open('G:/jiayou/Weibo.txt','r',encoding='utf-8') as f:
    while True:
        linestr = f.readline()#一个Event
        if not linestr:
            break
        linelist = linestr.split()#分隔符对字符串进行切片
        #print(linelist)
        eid = int(linelist[0][4:])#取第1个元素的第5位字符及以后
        label = int(linelist[1][-1:])#取第2个元素倒数第一个字符及以后
        result = []
        with open('G:/jiayou/pre1/%d.json'%(eid),'r',encoding='utf-8') as f1:
            print(eid)
            data = json.load(f1)
            fo1 = open('G:/jiayou/pre2/%d.json'%(eid),'w',encoding='utf-8')
            begin = int(data[0]["t"]) 
            end = int(data[len(data)-1]["t"])
            N=50
            num=1
            count=0
            interval = (end-begin)/N
            blog_length = positive_words = negative_words = sentiment_score=0
            url_sum = smiling_sum=frowning_sum=firstperson_num=hashtag_sum=mention_sum=0
            questin_sum=exclamation_sum=multi_question=multi_exclamation=0
            user_description=picture=verified=gender_m=city=0
            verified_type = []
            friends_count=followers_count=existtime=reputation=retweets=comments_count=0
            text = ""
            for i in range(len(data)):
                tem = {}
                time = int(data[i]["t"])
                if(time >= begin+interval*num):
                    num+=1
                    if(count!=0):
                        tem["text_list"] = text
                        tem["count"] = count
                        tem["length"]=blog_length/count
                        tem["positive_words"] = positive_words
                        tem["negative_words"] = negative_words
                        tem["sentiment_score"] = (positive_words - negative_words)/count
                        tem["url_sum"] = url_sum/count
                        tem["firstperson"] = firstperson_num/count
                        tem["hashtag"] = hashtag_sum/count
                        tem["mention"] = mention_sum/count
                        tem["questin_sum"] = questin_sum/count
                        tem["exclamation_sum"] = exclamation_sum/count
                        tem["multi_question"] = multi_question/count
                        tem["multi_exclamation"] = multi_exclamation/count
                        tem["user_description"] = user_description/count
                        tem["picture"] = picture/count
                        tem["verified"] = verified/count
                        result.append(tem)
                        count=1
                        blog_length = positive_words = negative_words = sentiment_score=0
                        url_sum = smiling_sum=frowning_sum=firstperson_num=hashtag_sum=mention_sum=0
                        questin_sum=exclamation_sum=multi_question=multi_exclamation=0
                        user_description=picture=verified=gender_m=city=0
                        verified_type = []
                        text = ""
                else:
                    count+=1
                    text += data[i]["text_list"]
                    blog_length += data[i]["length"]
                    positive_words += data[i]["positive_words"]
                    negative_words += data[i]["positive_words"]
                    url_sum += int(data[i]["url_sum"])
                    firstperson_num += data[i]["firstperson"]
                    hashtag_sum += data[i]["hashtag"]
                    mention_sum += data[i]["mention"]
                    questin_sum += data[i]["questin_sum"]
                    exclamation_sum += data[i]["exclamation_sum"]
                    multi_question += data[i]["multi_question"]
                    multi_exclamation += data[i]["multi_exclamation"]
                    user_description += data[i]["user_description"]
                    picture += data[i]["picture"]
                    verified += data[i]["verified"]
                    

            fo1.write(json.dumps(result,ensure_ascii=False,indent=4))
            fo1.close()
