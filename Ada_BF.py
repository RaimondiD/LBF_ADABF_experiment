import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import serialize
from Bloom_filter import hashfunc
from abstract_filter import Abstract_Filter


class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = []
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
    
    def insert(self, key, k):
        for j in range(int(k)):
            t = self.h[j](key)
            self.table[t] = 1
    def test(self, key, k):
        test_result = 0
        match = 0
        for j in range(int(k)):
            t = self.h[j](key)
            match += 1*(self.table[t] == 1)
        if match == k:
            test_result = 1
        return test_result

class OptimalAdaBloomFilter(Abstract_Filter):

    def __init__(self,opt_filter, opt_treshold, k_max):
        self.bloom_filter_opt = opt_filter
        self.thresholds_opt = opt_treshold
        self.k_max_opt = k_max
    
    def query(self, query_set):     
        negative_sample_test =  query_set
        ML_positive = negative_sample_test.iloc[:, 0][(negative_sample_test.iloc[:, -1] >= self.thresholds_opt[-2])]
        query_negative = negative_sample_test.iloc[:, 0][(negative_sample_test.iloc[:, -1] < self.thresholds_opt[-2])]
        score_negative = negative_sample_test.iloc[:, -1][(negative_sample_test.iloc[:, -1] < self.thresholds_opt[-2])]
        test_result = np.zeros(len(query_negative))
        ss = 0
        for score_s, query_s in zip(score_negative, query_negative):
            ix = min(np.where(score_s < self.thresholds_opt)[0])
            # thres = thresholds[ix]
            k = self.k_max_opt - ix
            test_result[ss] = self.bloom_filter_opt.test(query_s, k)
            ss += 1
        FP_items = sum(test_result) + len(ML_positive)
        return FP_items

def R_size(count_key, count_nonkey, R0):
    R = [0]*len(count_key)
    R[0] = R0
    for k in range(1, len(count_key)):
        R[k] = max(int(count_key[k] * (np.log(count_nonkey[0]/count_nonkey[k])/np.log(0.618) + R[0]/count_key[0])), 1)
    return R

def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]

    k_min = 0
    for k_max in range(num_group_min, num_group_max+1):
        for c in c_set:
            tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = np.zeros(k_max - k_min + 1)
            thresholds[-1] = 1.1
            num_negative = sum(train_negative.iloc[:, -1] <= thresholds[-1])
            num_piece = int(num_negative / tau) + 1
            score = train_negative.iloc[:, -1][(train_negative.iloc[:, -1] <= thresholds[-1])]
            score = np.sort(score)
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = score[score < thresholds[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]

            query = positive_sample.iloc[:, 0]
            score = positive_sample.iloc[:, -1]

            for score_s, query_s in zip(score, query):
                ix = min(np.where(score_s < thresholds)[0])
                k = k_max - ix
                bloom_filter.insert(query_s, k)
            ML_positive = train_negative.iloc[:, 0][(train_negative.iloc[:, -1] >= thresholds[-2])]
            query_negative = train_negative.iloc[:, 0][(train_negative.iloc[:, -1] < thresholds[-2])]
            score_negative = train_negative.iloc[:, -1][(train_negative.iloc[:, -1] < thresholds[-2])]

            test_result = np.zeros(len(query_negative))
            ss = 0
            for score_s, query_s in zip(score_negative, query_negative):
                ix = min(np.where(score_s < thresholds)[0])
                # thres = thresholds[ix]
                k = k_max - ix
                test_result[ss] = bloom_filter.test(query_s, k)
                ss += 1
            FP_items = sum(test_result) + len(ML_positive)
            print('False positive items: %d, Number of groups: %d, c = %f' %(FP_items, k_max, round(c, 2)))

            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max

    # print('Optimal FPs: %f, Optimal c: %f, Optimal num_group: %d' % (FP_opt, c_opt, num_group_opt))
    return OptimalAdaBloomFilter(bloom_filter_opt, thresholds_opt, k_max_opt)

'''
Implement Ada-BF
'''
def main(DATA_PATH,data_path_test, R_sum, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True, help="Minimum number of groups")
    parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True, help="Maximum number of groups")
    parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=True, help="minimum ratio of the keys")
    parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=True, help="maximum ratio of the keys")

    results = parser.parse_args(others)
    num_group_min = results.min_group
    num_group_max = results.max_group
    c_min = results.c_min
    c_max = results.c_max

    '''
    Load the data and select training data
    '''
    data = serialize.load_dataset(DATA_PATH)
    train_negative = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]

    '''
    Plot the distribution of scores
    '''
    plt.style.use('seaborn-deep')

    x = data.loc[data['label']== 1,'score']
    y = data.loc[data['label']== 0,'score']
    bins = np.linspace(0, 1, 25)

    plt.hist([x, y], bins, log=True, label=['Keys', 'non-Keys'])
    plt.legend(loc='upper right')
    plt.savefig('./Score_Dist.png')
    plt.show()

    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    opt_Ada = Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample)
    
    '''Stage 2: Run Ada-BF on all the samples'''
    ### Test Queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True, help="size of the Ada-BF")
    result =parser.parse_known_args() 
    main(result[0].data_path, result[0].R_sum, result[1])

