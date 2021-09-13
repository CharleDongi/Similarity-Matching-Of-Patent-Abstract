import csv
import numpy
from bert_serving.client import BertClient
import numpy as np

class SimilarModel:
    def __init__(self):
        # ip默认为本地模式，如果bert服务部署在其他服务器上，修改为对应ip
        self.bert_client = BertClient()

    def close_bert(self):
        self.bert_client.close()

    def get_sentence_vec(self,sentence):
        '''
        根据bert获取句子向量
        :param sentence:
        :return:
        '''
        return self.bert_client.encode([sentence])[0]

    def cos_similar(self,sen_a_vec, sen_b_vec):
        '''
        计算两个句子的余弦相似度
        :param sen_a_vec:
        :param sen_b_vec:
        :return:
        '''
        vector_a = np.mat(sen_a_vec)
        vector_b = np.mat(sen_b_vec)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        return cos

if __name__=='__main__':
    # 1.按行读取文本内容，并且写入文件，仅执行一次！
    bert_client = SimilarModel()
    # file = open('data.text', 'a+', encoding='utf-8')
    # csv_writer = csv.writer(file)
    a = []
    file = open('test.txt', 'r', encoding="utf-8")
    i = 0
    while 1:
        line = file.readline()
        if not line:
            break
        sentence_b_vec = bert_client.get_sentence_vec(line)
        a.append(sentence_b_vec)
        i =i+1
        print("成功：", i , "次！")
        # csv_writer.writerow(sentence_b_vec)
    # file.close()
    np.save("vector.npy", a)


