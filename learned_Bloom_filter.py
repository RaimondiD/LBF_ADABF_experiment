import numpy as np
import pandas as pd
import argparse
from Bloom_filter import BloomFilter



def Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample):
    FP_opt = train_negative.shape[0]

    for threshold in np.arange(min_thres, max_thres+10**(-6), 0.01):
        query = positive_sample.loc[(positive_sample['score'] <= threshold),'url']
        n = len(query)
        print(n)
        bloom_filter = BloomFilter(n, R_sum)
        bloom_filter.insert(query)
        ML_positive = train_negative.loc[(train_negative['score'] > threshold),'url']
        bloom_negative = train_negative.loc[(train_negative['score'] <= threshold),'url']
        BF_positive = bloom_filter.test(bloom_negative, single_key=False)
        FP_items = sum(BF_positive) + len(ML_positive)

        print('Threshold: %f, False positive items: %d' %(round(threshold, 2), FP_items))
        if FP_opt > FP_items:
            FP_opt = FP_items
            thres_opt = threshold
            bloom_filter_opt = bloom_filter
    return bloom_filter_opt, thres_opt





def main(data_path, size_filter, other):
    parser = argparse.ArgumentParser()
    
    

    parser.add_argument('--threshold_min', action="store", dest="min_thres", type=float, required=True,
                    help="Minimum threshold for positive samples")
    parser.add_argument('--threshold_max', action="store", dest="max_thres", type=float, required=True,
                    help="Maximum threshold for positive samples")
    

    results = parser.parse_args(other)
    DATA_PATH = data_path
    min_thres = results.min_thres
    max_thres = results.max_thres
    R_sum = size_filter

    '''
    Load the data and select training data
    '''
    data = pd.read_csv(DATA_PATH)
    negative_sample = data.loc[(data['label']==0)]
    positive_sample = data.loc[(data['label']==1)]
    train_negative = negative_sample.sample(frac = 0.3)

    
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    bloom_filter_opt, thres_opt = Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample)

    '''Stage 2: Run LBF on all the samples'''
    ### Test queries
    ML_positive = negative_sample.loc[(negative_sample['score'] > thres_opt), 'url']
    bloom_negative = negative_sample.loc[(negative_sample['score'] <= thres_opt), 'url']
    score_negative = negative_sample.loc[(negative_sample['score'] < thres_opt), 'score']
    BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
    FP_items = sum(BF_positive) + len(ML_positive)
    FPR = FP_items/len(negative_sample)
    #print('False positive items: {}; FPR: {}; Size of queries: {}'.format(FP_items, FPR, len(negative_sample)))
    return FP_items,FPR, len(negative_sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
    result =parser.parse_known_args() 
    print(result[0])
    main(result[0].data_path,result[1])

