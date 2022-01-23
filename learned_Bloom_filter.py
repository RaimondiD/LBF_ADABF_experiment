import numpy as np
import pandas as pd
import argparse
import time
import serialize
from Bloom_filter import BloomFilter

def Find_Optimal_Parameters(R_sum, train_negative, positive_sample, quantile_order):
    FP_opt = train_negative.shape[0]

    # Calcolo soglie da testare
    train_dataset = np.array(pd.concat([train_negative, positive_sample]).iloc[:, -1]) # 30 % negativi + tutte le chiavi
    thresholds_list = [np.quantile(train_dataset, i * (1 / quantile_order)) for i in range(1, quantile_order)] if quantile_order < len(train_dataset) else np.sort(train_dataset)
    print(f"Quantili {[i * (1 / quantile_order) for i in range(1, quantile_order)]}, Soglie: {thresholds_list}")

    for threshold in thresholds_list:
        query = positive_sample.iloc[:, 1][(positive_sample.iloc[:, -1] <= threshold)]
        n = len(query)
        bloom_filter = BloomFilter(n, R_sum)
        bloom_filter.insert(query)
        ML_positive = train_negative.iloc[:, 1][(train_negative.iloc[:, -1] > threshold)]
        bloom_negative = train_negative.iloc[:, 1][(train_negative.iloc[:, -1] <= threshold)]
        BF_positive = bloom_filter.test(bloom_negative, single_key = False)
        FP_items = sum(BF_positive) + len(ML_positive)

        print('Threshold: %f, False positive items: %d' %(round(threshold, 2), FP_items))
        if FP_opt > FP_items:
            FP_opt = FP_items
            thres_opt = threshold
            bloom_filter_opt = bloom_filter

    return bloom_filter_opt, thres_opt

def main(data_path, data_path_test, size_filter, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresholds_q', action = "store", dest = "thresholds_q", type = int, required = True, help = "order of quantiles to be tested")
    results = parser.parse_args(others)

    thresholds_q = results.thresholds_q
    DATA_PATH = data_path
    R_sum = size_filter

    '''
    Load the data and select training data.
    '''
    data = serialize.load_dataset(DATA_PATH)
    train_negative = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    bloom_filter_opt, thres_opt = Find_Optimal_Parameters(R_sum, train_negative, positive_sample, thresholds_q)
    '''Stage 2: Run LBF on all the samples'''
    ### Test queries
    negative_sample_test = serialize.load_dataset(data_path_test)
    start = time.time()
    ML_positive = negative_sample_test.iloc[:, 1][(negative_sample_test.iloc[:, -1] > thres_opt)]
    bloom_negative = negative_sample_test.iloc[:, 1][(negative_sample_test.iloc[:, -1] <= thres_opt)]
    BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
    end = time.time()
    FP_items = sum(BF_positive) + len(ML_positive)
    FPR = FP_items/len(negative_sample_test)
    #print('False positive items: {}; FPR: {}; Size of queries: {}'.format(FP_items, FPR, len(negative_sample)))
    return FP_items,FPR, len(negative_sample_test), (end-start)/len(negative_sample_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    result =parser.parse_known_args() 
    print(result[0])
    main(result[0].data_path, result[1])

