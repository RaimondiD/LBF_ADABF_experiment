import numpy as np
import argparse
from Bloom_filter import BloomFilter
from abstract_filter import Abstract_Filter
import serialize

class PartitionedBloomFilter(Abstract_Filter):
    def __init__(self, bloom_filters_opt, thresholds_opt):
        self.bloom_filter_opt = bloom_filters_opt
        self.thresholds_opt = thresholds_opt

    def query(self, query_set):
        ML_positive = query_set.iloc[:, 1][(query_set['score'] >= self.thresholds_opt[-2])]
        query_negative = query_set.iloc[:, 1][(query_set['score'] < self.thresholds_opt[-2])]
        score_negative = query_set.iloc[:, -1][(query_set['score'] < self.thresholds_opt[-2])]
        test_result = np.zeros(len(query_negative))
        ss = 0
        for score_s, query_s in zip(score_negative, query_negative):
            ix = min(np.where(score_s < self.thresholds_opt)[0]) - 1
            test_result[ss] = self.bloom_filter_opt[ix].test(query_s)
            ss += 1
        FP_items = sum(test_result) + len(ML_positive)

        return FP_items

def DP_KL_table(train_negative, positive_sample, num_group_max):
    negative_score = train_negative['score']
    positive_score = positive_sample['score']
    interval = 1/10000
    min_score = min(np.min(positive_score), np.min(negative_score))
    max_score = min(np.max(positive_score), np.max(negative_score))
    score_partition = np.arange(min_score-10**(-10),max_score+10**(-10)+interval,interval)

    h = [np.sum((score_low<=negative_score) & (negative_score<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
    h = np.array(h)
    ## Merge the interval with less than 5 nonkey
    delete_ix = []
    for i in range(len(h)):
        if h[i] < 5:
            delete_ix += [i]
    score_partition = np.delete(score_partition, [i for i in delete_ix])
    ## Find the counts in each interval
    h = [np.sum((score_low<=negative_score) & (negative_score<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
    h = np.array(h)
    g = [np.sum((score_low<=positive_score) & (positive_score<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
    g = np.array(g)

    ## Merge the interval with less than 5 keys
    delete_ix = []
    for i in range(len(g)):
        if g[i] < 5:
            delete_ix += [i]
    score_partition = np.delete(score_partition, [i+1 for i in delete_ix])

    ## Find the counts in each interval
    h = [np.sum((score_low<=negative_score) & (negative_score<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
    h = np.array(h)
    g = [np.sum((score_low<=positive_score) & (positive_score<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
    g = np.array(g)
    
    g = g/np.sum(g)
    h = h/np.sum(h)
    n = len(score_partition)
    k = num_group_max
    optim_KL = np.zeros((n,k))
    optim_partition = [[0]*k for _ in range(n)]

    for i in range(n):
        optim_KL[i,0] = np.sum(g[:(i+1)]) * np.log2(sum(g[:(i+1)])/sum(h[:(i+1)]))
        optim_partition[i][0] = [i]

    for j in range(1,k):
        for m in range(j,n):
            candidate_par = np.array([optim_KL[i][j-1]+np.sum(g[i:(m+1)])*np.log2(np.sum(g[i:(m+1)])/np.sum(h[i:(m+1)])) for i in range(j-1,m)])
            optim_KL[m][j] = np.max(candidate_par)
            ix = np.where(candidate_par == np.max(candidate_par))[0][0] + (j-1)
            if j > 1:
                optim_partition[m][j] = optim_partition[ix][j-1] + [ix]
            else:
                optim_partition[m][j] = [ix]   
    return optim_partition, score_partition

def Find_Optimal_Parameters(num_group_min, num_group_max, R_sum, train_negative, positive_sample, negative_sample, optim_partition, score_partition):
    FP_opt = train_negative.shape[0]
    negative_score = negative_sample['score']
    positive_score = positive_sample['score']

    for num_group in range(num_group_min, num_group_max+1):
        ### Determine the thresholds    
        thresholds = np.zeros(num_group + 1)
        thresholds[0] = -0.1
        thresholds[-1] = 1.1
        inter_thresholds_ix = optim_partition[-1][num_group-1]
        inter_thresholds = score_partition[inter_thresholds_ix]
        thresholds[1:-1] = inter_thresholds

        ### Count the keys of each group
        query = positive_sample.iloc[:, 1]
        score = positive_sample['score']
        
        count_nonkey = np.zeros(num_group)
        count_key = np.zeros(num_group)
        query_group = []
        bloom_filter = []
        for j in range(num_group):
            count_nonkey[j] = sum((negative_score >= thresholds[j]) & (negative_score < thresholds[j + 1]))
            count_key[j] = sum((positive_score >= thresholds[j]) & (positive_score < thresholds[j + 1]))
            query_group.append(query[(score >= thresholds[j]) & (score < thresholds[j + 1])])


        ### Search the Bloom filters' size
        def R_size(c):
            R = 0
            for j in range(len(count_key)-1):
                R += max(1, count_key[j]/np.log(0.618)*(np.log(count_key[j]/count_nonkey[j])+c))
            return R
        
        lo=-100
        hi=0
        while abs(lo-hi) > 10**(-3):
            mid = (lo+hi)/2
            midval = R_size(mid)
            if midval < R_sum:
                hi = mid
            elif midval >= R_sum: 
                lo = mid
        c = mid

        R = np.zeros(num_group)
        for j in range(num_group-1):
            R[j] = int(max(1, count_key[j]/np.log(0.618)*(np.log(count_key[j]/count_nonkey[j])+c)))
        
        # print(count_key, R)
        Bloom_Filters = []
        for j in range(int(num_group - 1)):
            if count_key[j]==0:
                Bloom_Filters.append([0])
            else:
                Bloom_Filters.append(BloomFilter(count_key[j], R[j]))
                Bloom_Filters[j].insert(query_group[j])

        ### Test querys
        ML_positive = train_negative.iloc[:, 1][(train_negative['score'] >= thresholds[-2])]
        query_negative = train_negative.iloc[:, 1][(train_negative['score'] < thresholds[-2])]
        score_negative = train_negative.iloc[:, -1][(train_negative['score'] < thresholds[-2])]

        test_result = np.zeros(len(query_negative))
        ss = 0
        for score_s, query_s in zip(score_negative, query_negative):
            ix = min(np.where(score_s < thresholds)[0]) - 1
            test_result[ss] = Bloom_Filters[ix].test(query_s)
            ss += 1
        
        FP_items = sum(test_result) + len(ML_positive)
        FPR = FP_items/len(train_negative)
        print('False positive items: {}, FPR: {} Number of groups: {}'.format(FP_items, FPR, num_group))
        if FP_opt > FP_items:
            FP_opt = FP_items
            Bloom_Filters_opt = Bloom_Filters
            thresholds_opt = thresholds

    return PartitionedBloomFilter(Bloom_Filters_opt, thresholds_opt)

def main(data_path, size_filter, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True,  help="Minimum number of groups")
    parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True, help="Maximum number of groups")

    results = parser.parse_args(others)
    num_group_min = results.min_group
    num_group_max = results.max_group
    data = serialize.load_dataset(data_path)
    train_negative = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]

    optim_partition, score_partition = DP_KL_table(train_negative, positive_sample, num_group_max)
    opt_Plbf = Find_Optimal_Parameters(num_group_min, num_group_max, size_filter, train_negative, positive_sample, train_negative, optim_partition, score_partition)

    return opt_Plbf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True, help="size of the Ada-BF")
    result =parser.parse_known_args() 

    main(result[0].data_path, result[0].R_sum, result[1])