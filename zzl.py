#%% 

# 所有的数据分为一下几类：
# 诊断复杂，中有数字，形容描述 如 0013


# 

import jieba
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cla


def getsample( filename , featurestr,loopnumber):
    fo = open(filename)
    diagnose = list()
    # print ("文件名为: ", fo.name)
    for i in range(loopnumber):
        line = fo.readline()
        # print(line)
        str1 = line.split('$')
        if str1[1] == featurestr:
            dia = str1[2].split('\n') # remove '\n'
            if dia[0] not in diagnose:
                diagnose.append(dia[0])
    return diagnose

def getsimma(da):
    
    size = len(da)
    all_doc_list = []
    for doc in da:
        doc_list = [word for word in jieba.cut(doc)]
        all_doc_list.append(doc_list)

    dictionary = corpora.Dictionary(all_doc_list)
    corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

    sim = np.zeros((size,size))
    for i in range(size):
        doc_test_vec = dictionary.doc2bow(all_doc_list[i])
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim[i,:] = 1/(np.array(index[tfidf[doc_test_vec]])+0.01)
    
    return sim

def main():
    
    # 读取的文件名
    filename = "meinian_round1_data_part1_20180408.txt"
    # 读取的条目名
    featurestr = '0516'

    da = getsample( filename , featurestr, 30000) # 提取不同的信息
    numoftxt =len(da) # 获得不同的信息的个数
    sim = getsimma(da) # 获得相似度
    out = cla.main(sim,5,numoftxt) # 分类模型的建立
    
    # 生成对应的字典
    # dist = dist()
    # dist.append()



    # 打印所有的条目
    for i in range(numoftxt):
        # if out[i,0] == claser:
        print(da[i])

    # 打印各个分类
    for claser in range(5):
        print('--------------------------------------')
        for i in range(numoftxt):
            if out[i,0] == claser:
                print(da[i])

    # print(all_doc_list)



if __name__ == '__main__':
    main()
