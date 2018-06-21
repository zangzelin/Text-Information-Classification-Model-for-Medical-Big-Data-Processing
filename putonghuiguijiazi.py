
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:30:16 2018

@author: yang
"""
from imp import reload
import sys
reload(sys)
# sys.setdefaultencoding("utf-8")
import gc
import re
import sys
import time
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os.path
import os
import re
import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import lightgbm as lgb
import gensim
from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
train    = pd.read_csv('data/train_set.csv')#1653
test     = pd.read_csv('data/test_set.csv')



train_temp=train.copy()
test_temp=test.copy()

def chushuang(x):
    if type(x)==type('str'):
        return x.split(' ')[0]
    else:

        return x
for i,j in enumerate(['2403','2405']):
    train[j]=train[j].map(chushuang)
    test[j]=test[j].map(chushuang)
#train['2403']=train['2403'].str.split(' ')[0]
le=LabelEncoder()
train=train.convert_objects(convert_numeric=True)
test=test.convert_objects(convert_numeric=True)
#train=train.loc[0:100,:]
#test=test.loc[0:100,:]
def remain_float(df,thresh=0.96):
    float_feats = []
    str_feats = []
    print('----------select float data-----------')
    print('sum',len(df.columns))

    for c in df.columns:
        num_missing = df[c].isnull().sum()
        missing_percent = num_missing / float(df.shape[0])
        if df[c].dtype == 'float64' and missing_percent<thresh:
            float_feats.append(c)
        elif df[c].dtype == 'int64':
            print(c)
        else:
            str_feats.append(c)


    return float_feats,str_feats


floatyin_feats=['1850','2177','2371','2376','300017','300036','809021']
float_feats,str_feats=remain_float(train,thresh=0.96)
str_feats=str_feats+floatyin_feats
str_feats.remove('vid')
train_float=train[float_feats]
test_float=test[float_feats]
train_str=train_temp[str_feats]
test_str=test_temp[str_feats]
a=train_float.head(100)


#-------------------str kaishi
str_feature=[c for c in train_str.columns if c not in floatyin_feats]
train_xue=pd.DataFrame()
test_xue=pd.DataFrame()

#----------nannv
def zigong(x):
    if type(x)==type('str'):
        return 1
    else:

        return 0
for i,j in enumerate(['0121']):
    train_xue['sex']=train_str[j].map(zigong)
    test_xue['sex']=test[j].map(zigong)
def qianliexian(x):
    if type(x)==type('str'):
        return 2
    else:

        return 0
for i,j in enumerate(['0120']):
    train_xue['sex']=train_xue['sex'] | train_str[j].map(qianliexian)
    test_xue['sex']=test_xue['sex'] | test[j].map(qianliexian)

str_feature=[c for c in train_str.columns if c not in floatyin_feats]
train_str=train_str.fillna("正常")
test_str=test_str.fillna("正常")
def zigongruxian0102(x):
    if '子宫' in x or '附件' in x:
        return 1
    elif '前列腺' in x:
        return 2
    else:
        return 0
for i,j in enumerate(['0101','0102']):
    train_xue['sex']=train_xue['sex'] | train_str[j].map(zigongruxian0102)
    test_xue['sex']=test_xue['sex'] | test_str[j].map(zigongruxian0102)
train_xue = pd.concat([train_xue, pd.get_dummies(train_xue['sex'])], axis=1)
test_xue = pd.concat([test_xue, pd.get_dummies(test_xue['sex'])], axis=1)
train_xue = train_xue.drop('sex',axis=1)
test_xue = test_xue.drop('sex',axis=1)
train_xue.columns=['man','woman','el']
test_xue.columns=['man','woman','el']

train_str=train_str.fillna("正常")
test_str=test_str.fillna("正常")
#----------------------xue ya zheng ze
xue_feature=['血压','糖尿病','冠心病','甲肝','病史','治疗','胃溃疡','房颤','间断','痛风','血糖','冠心病','胃炎','结石',
               '血吸虫','肺心病','甲亢','心肌炎','脑血栓','尿酸','肝硬化','血脂','血症','肾炎','肥胖',
              '胰腺炎','脂肪','动脉硬化','动脉壁','血管壁','心动过缓','心动过速','甲状腺功能亢进','甲状腺病变','乙肝']
for i,j in enumerate(str_feature):
    if i==0:
        for name in xue_feature:
            train_xue[name] = train_str[j].map(lambda x:1 if name in x else 0)
            test_xue[name] = test_str[j].map(lambda x:1 if name in x else 0)
            #train_xue['血压量']=train_str[j].map(xueya)
            #test_xue['血压量']=test_str[j].map(xueya)
    else:
        for name in xue_feature:
            train_xue[name]=train_xue[name] | train_str[j].map(lambda x:1 if name in x else 0)
            test_xue[name] = test_xue[name] | test_str[j].map(lambda x:1 if name in x else 0)
            #train_xue['血压量']=train_xue['血压量'] | train_str[j].map(xueya)
            #test_xue['血压量'] = test_xue['血压量'] | test_str[j].map(xueya)

train_xue['动脉壁']=train_xue['动脉硬化']|train_xue['动脉壁']
train_xue['甲亢']=train_xue['甲亢']|train_xue['甲状腺功能亢进']|train_xue['甲状腺病变']
del(train_xue['动脉硬化'])
del(train_xue['甲状腺功能亢进'])
del(train_xue['甲状腺病变'])



#--------str to shuzi tezheng
def chang(x):
    if type(x)==type('str'):
        return len(x)
    else:
        return np.nan

#train_temp[float_feats].applymap(dd)
#print pd.(train_temp[float_feats].values.flatten())
#train_temp[float_feats].str.isalnum()

for i,j in enumerate(train_temp[str_feats].columns):
    name=str_feats[i]+'_0'
    if 1:
        train_xue[name] = train_temp[j].map(chang)
        test_xue[name] = test_temp[j].map(chang)
train_xue=train_xue.fillna(train_xue.median())
test_xue=test_xue.fillna(test_xue.median())
for i,j in enumerate(train_temp[str_feats].columns):
    name=str_feats[i]+'_0'
    if 1:
        train_xue[name] = (train_xue[name]-train_xue[name].mean())/train_xue[name].std()
        test_xue[name] = (test_xue[name] - test_xue[name].mean()) / test_xue[name].std()

#----------yinxing
train_yin=pd.DataFrame()
test_yin=pd.DataFrame()
def xuanze(x):
    #if re.compile(r'[0-9]').match(x):
   #     return 3
   # elif '+-' in x:
   #     return 1
   # elif '+' or '阳性' in x:
   #     return 2
    x=str(x)
    if '阴性' in x or '正常' in x or 'Normal' in x or '未做' in x or '-' == x or '(-)' in x:
        return 0
    else:
        return 1
yinxing_feats=['1850','2177','2371','2376','300017','300036','809021','3207','3429','3430','3485','3486','3730','669024','3197',
               '3196','3195','3194','3192','3191','3190','300044','300019','300018',
               '300005','2282','2233','1363','2231','100010','2231','2230','2229','2228']
#'1850','2177','2371','2376','300017','300036','809021','360','3193','3189',
for i,j in enumerate(yinxing_feats):
    name=yinxing_feats[i]+'_1'
    if 1:
        pass
        train_yin[name]=train_str[j].map(xuanze)
        test_yin[name] = test_str[j].map(xuanze)



#----------zhao
#for i,j in enumerate(str_feats):
   # print j
   # print train_str[j][train_str[j].str.contains('眼压')]
def test(x):
    x=str(x)
    if '眼压' in x:
        return True
    else:
        return False




#for i,j in enumerate(str_feats):
   # if i==0:
   #  #   test_xue['xueya'] = test_str[j].map(xueya)
    #else:
    #    train_xue['xueya']=train_xue['xueya'] | train_str[j].map(xueya)
    #    test_xue['xueya'] = test_xue['xueya'] | test_str[j].map(xueya)


c=0
#for i in train_temp[str_feats].columns:
 #   if train_temp[i].dtype=='object':
 #       print train_temp[i][train_temp[i].str.contains("阴性")==True]


cw = lambda x: list(jieba.cut(x))
def cw1(x):
    #print x.shape
    x=str(x)
    temp=list(jieba.cut(x))
    temp=' '.join(temp)
    return temp
train_str=train_str.fillna("正常")
test_str=test_str.fillna("正常")
train_vector=pd.DataFrame()
test_vector=pd.DataFrame()
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
for i,j in enumerate(train_str.columns[0:]):
    print(j)
    name="vec1_"+str(i)
    train_vector[name] = train_str[j].apply(cw1)
    X = vectorizer.fit_transform(train_vector[name])
    tfidf = transformer.fit_transform(X)
    tfidf=tfidf.toarray()
    train_vector[name]=np.sum(tfidf,axis=1)

    test_vector[name] = test_str[j].apply(cw1)
    X = vectorizer.fit_transform(test_vector[name])
    tfidf = transformer.fit_transform(X)
    tfidf = tfidf.toarray()
    test_vector[name] = np.sum(tfidf, axis=1)

#----------he qi lai
train_float=pd.concat([train_float,train_xue],axis=1)
test_float=pd.concat([test_float,test_xue],axis=1)

train_float=pd.concat([train_float,train_yin],axis=1)
test_float=pd.concat([test_float,test_yin],axis=1)

train_float=pd.concat([train_float,train_vector],axis=1)
test_float=pd.concat([test_float,test_vector],axis=1)
#vector.columns= ["vec_{0}".format(i) for i in range(0,1)]
'''
for i,j in enumerate(vector.columns[0:1]):
    if i==0:
        d2v_train=vector[j]
    else:
        d2v_train+=vector[j]
'''
#model = Word2Vec(d2v_train,size=n,window=4,min_count=1,negative=3,
#                 sg=1,sample=0.001,hs=1,workers=4)      
#d2v_train = pd.concat(vector.columns, ignore_index = True) 

#-------------------yi chang zhi

def clean_label_shousuo(x):
    if x<90:
        x=90
    if x>200:
        x=200

    return x
def clean_label_shuzhang(x):
    if x<55:
        x=55
    if x>125:
        x=125

    return x
def clean_label_di(x):
    if x<0:
        x=0

    return x
train_float[train_float.columns[0]]=train_float[train_float.columns[0]].apply(clean_label_shousuo)
train_float[train_float.columns[1]]=train_float[train_float.columns[1]].apply(clean_label_shuzhang)
train_float[train_float.columns[4]]=train_float[train_float.columns[4]].apply(clean_label_di)

#-------------- bu median
train_float=train_float.fillna(train_float.median())
test_float=test_float.fillna(train_float.median())

#-------- start
result=pd.read_csv('./data/meinian_round1_test_a_20180409.csv')
nowTime = datetime.datetime.now().strftime('%m%d%H%M')  # 现在
name = 'lgb_' + nowTime
score=list()



for q in range(5):

    X_train, X_val, y_train, y_val = train_test_split(train_float[train_float.columns[5:]], train_float[train_float.columns[q]],test_size=0.2, random_state=1)
    train = lgb.Dataset(X_train, label=y_train)
    val  = lgb.Dataset(X_val, label=y_val,reference=train)

    def binary_error(preds, train_data):
        labels = train_data.get_label()
        e=np.mean(np.square(np.log10(preds+1)-np.log10(labels+1)))
        return 'myloss',e,False
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        #'metric': 'l2',
        #'objective': 'multiclass',
        #'metric': 'multi_error',
        #'num_class':5,
        'min_child_weight': 3,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'learning_rate': 0.005,
        'tree_method': 'exact',
        'seed': 2017,
        'nthread': 12,
        'silent': True
        }


    num_round = 30000
    gbm = lgb.train(params,
                      train,
                      num_round,
                      verbose_eval=50,
                      valid_sets=[val],
                      feval=binary_error,
                      early_stopping_rounds=1500
                      )

    score.append(gbm.best_score['valid_0']['myloss'])

    preds_sub=gbm.predict(test_float[test_float.columns[5:]])

    gain = gbm.feature_importance('gain')
    ft = pd.DataFrame({'feature': gbm.feature_name(), 'split': gbm.feature_importance('split'),
                       'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft)

    result.iloc[:,q+1] = preds_sub
strresult="-"
for i in score:
    strresult+=str('%.5f-' % i)


#test_float=test_float.iloc[:,0:6]
result.to_csv(name+str(sum(score))+strresult+'.csv', index=False,header=None)


