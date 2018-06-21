import jieba
from gensim import corpora,models,similarities

doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc_test="我喜欢上海的小吃"


all_doc = []
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)
all_doc.append(doc5)
all_doc.append(doc6)
all_doc.append(doc7)
print('222')
# print(all_doc)

all_doc_list = []
for doc in all_doc:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

# print(all_doc_list)



doc_test_list = [word for word in jieba.cut(all_doc[1])]
doc_test_list

dictionary = corpora.Dictionary(all_doc_list)

dictionary.keys()
dictionary.token2id
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

doc_test_vec = dictionary.doc2bow(doc_test_list)
# print(doc_test_vec)


tfidf = models.TfidfModel(corpus)
# print(tfidf[doc_test_vec])

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]
print(sim)

# print(sorted(enumerate(sim), key=lambda item: -item[1]))