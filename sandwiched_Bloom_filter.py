from operator import pos
import numpy as np
import pandas as pd
import argparse
import math
import time
import serialize
from Bloom_filter import BloomFilter


def Find_Optimal_Parameters(b, train_negative, positive_sample, quantile_order = 10):
    '''
    B1 = Initial Bloom filter
    B2 = Backup Bloom filter
    m = |K|
    
    Assumendo di avere b*m budget bits
    b2* = FN log_alpha( FP/( (1-FP)*(1\(FN - 1)) ) )    Taglia filtro di backup
    b1* = b - b2*                                       Taglia filtro iniziale
    Dove FP indica il tasso di falsi positivi del classificatore su un insieme di non chiavi, e FN il tasso di falsi negativi del classificatore.
    '''
    if b < 1: 
        print("err")
        return 
    FP_opt = train_negative.shape[0] + 1
    # Calcolo soglie da testare
    train_dataset = np.array(pd.concat([train_negative, positive_sample])['score']) # 30 % negativi + tutte le chiavi
    thresholds_list = [np.quantile(train_dataset, i * (1 / quantile_order)) for i in range(1, quantile_order)] if quantile_order < len(train_dataset) else np.sort(train_dataset)

    key_B1 = positive_sample
    keyB1_len = len(key_B1)
    m = len(positive_sample)

    for threshold in thresholds_list:
        ML_false_positive = (train_negative.iloc[:, -1] > threshold) # maschera falsi positivi generati dal modello rispetto alla soglia considerata,
        ML_true_negative = (train_negative.iloc[:, -1] <= threshold) # maschera veri negativi generati dal modello rispetto alla soglia considerata
        ML_false_negative = (positive_sample.iloc[:, -1] <= threshold) # maschera falsi negativi generati dal modello rispetto alla soglia considerata

        FP = (train_negative[ML_false_positive].iloc[:, 1].size) / train_negative.iloc[:, 1].size # stima probabilità di un falso positivo dal modello
        FN = (positive_sample[ML_false_negative].iloc[:, 1].size) / positive_sample.iloc[:, 1].size #stima probabilità di un falso negativo dal modello
        if (FP == 0.0 or FP == 1.0) or (FN == 1.0 or FN == 0.0): continue

        b2 = FN * math.log(FP / ((1 - FP) * ((1/FN) - 1)), 0.6185)
        b1 = b - b2
        if b1 <= 0: # Non serve avere SLBF
            print("b1 = 0")
            break
        
        print(f"FP: {FP}, FN: {FN}, b: {b}, b1: {b1}, b2: {b2}")
        
        # Creazione filtro iniziale
        B1 = BloomFilter(keyB1_len, b1*m)
        B1.insert(key_B1.iloc[:, 1])
        # Creazione filtro backup
        key_b2 = positive_sample[ML_false_negative].iloc[:, 1]
        B2 = BloomFilter(len(key_b2), b2*m)
        B2.insert(key_b2)
        # Calcolo FPR
        B1_false_positive = B1.test(train_negative.iloc[:, 1], single_key = False)
        ML_false_positive_list = train_negative.iloc[:, 1][(B1_false_positive) & (ML_false_positive)]
        ML_true_negative_list = train_negative.iloc[:, 1][(B1_false_positive) & (ML_true_negative)]
        B2_false_positive = B2.test(ML_true_negative_list, single_key = False)
        total_false_positive = sum(B2_false_positive) + len(ML_false_positive_list)
        print('Threshold: %f, False positive items: %d' %(round(threshold, 3), total_false_positive))
        if FP_opt > total_false_positive:
            FP_opt = total_false_positive
            optimal_threshold = threshold
            optimal_B1 = B1
            optimal_B2 = B2

    return optimal_B1, optimal_B2, optimal_threshold
    
def main(DATA_PATH_train, DATA_PATH_test, R_sum, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresholds_q', action = "store", dest = "thresholds_q", type = int, required = True, help = "order of quantiles to be tested")
    results = parser.parse_args(others)

    thresholds_q = results.thresholds_q

    '''
    Load the data and select training data
    '''
    data = serialize.load_dataset(DATA_PATH_train)
    train_negative_sample = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    b = R_sum / len(positive_sample)

    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    optimal_B1, optimal_B2, thres_opt = Find_Optimal_Parameters(b, train_negative_sample, positive_sample, thresholds_q)

    '''Stage 2: Run SLBF on all the samples'''
    ### Test queries
    negative_sample_test = serialize.load_dataset(DATA_PATH_test)
    start = time.time()
    B1FalsePositive = optimal_B1.test(negative_sample_test.iloc[:, 1], single_key = False)
    FP_ML = negative_sample_test.iloc[:, 1][(B1FalsePositive == 1) & (negative_sample_test['score'] > thres_opt)]
    Negative_ML = negative_sample_test.iloc[:, 1][(B1FalsePositive == 1) & (negative_sample_test['score'] <= thres_opt)]
    BF_positive = optimal_B2.test(Negative_ML, single_key = False)
    end = time.time()
    FP_items = sum(BF_positive) + len(FP_ML)
    FPR = FP_items/len(negative_sample_test)
    print('False positive items: {}; FPR: {}; Size of queries: {}'.format(FP_items, FPR, len(negative_sample_test)))
    return FP_items, FPR, len(negative_sample_test), (end-start)/len(negative_sample_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = "store", dest = "data_path", type = str, required = True, help = "path of the dataset")
    parser.add_argument('--size_of_Sandwiched', action = "store", dest = "R_sum", type = int, required = True, help = "size of the Ada-BF")
    result =parser.parse_known_args()  
    main(result[0].data_path, result[0].R_sum, result[1])


