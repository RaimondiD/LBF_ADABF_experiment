import numpy as np
import pandas as pd
import argparse
import math
from Bloom_filter import BloomFilter

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--threshold_min', action="store", dest="min_thres", type=float, required=True,
                    help="Minimum threshold for positive samples")
parser.add_argument('--threshold_max', action="store", dest="max_thres", type=float, required=True,
                    help="Maximum threshold for positive samples")
parser.add_argument('--bits_per_key', action="store", dest="b", type=float, required=True,
                    help="bits per key")

results = parser.parse_args()
DATA_PATH = results.data_path
min_thres = results.min_thres
max_thres = results.max_thres
R_sum = results.b

'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
negative_sample = data.loc[(data['label']==-1)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)

def Find_Optimal_Parameters(max_thres, min_thres, b, train_negative, positive_sample):
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
    # FP e FN  posso calcolarli così perché ho assunzione che le train negative sia distribuito come l'insieme delle chiavi, ma se non è così?
    FP_opt = train_negative.shape[0]
    for threshold in np.arange(min_thres, max_thres+10**(-6), 0.01):
        FP = (train_negative['url'][(train_negative['score'] > threshold)].size) / train_negative['url'].size
        FN = (positive_sample['url'][(positive_sample['score'] <= threshold)].size) / positive_sample['url'].size
        b2 = FN * math.log(FP / ((1 - FP) * ((1/FN) - 1)), 0.6185)
        b1 = b - b2
        if b1 <= 0: # Non serve avere SLBF
            print("b1 = 0")
            continue
        m = len(positive_sample)
        print(f"FP: {FP}, FN: {FN}, b: {b}, b1: {b1}, b2: {b2}")
        
        # Creazione filtro iniziale
        KeyB1 = positive_sample
        B1 = BloomFilter(len(KeyB1), b1*m)
        B1.insert(KeyB1['url'])
        # Creazione filtro backup
        KeyB2 = positive_sample['url'][(positive_sample['score'] <= threshold)]
        B2 = BloomFilter(len(KeyB2), b2*m)
        B2.insert(KeyB2)
        # Calcolo FPR
        B1FalsePositive = B1.test(train_negative['url'], single_key = False)
        FP_ML = train_negative['url'][(B1FalsePositive == 1) & (train_negative['score'] > threshold)]
        Negative_ML = train_negative['url'][(B1FalsePositive == 1) & (train_negative['score'] <= threshold)]
        FP_B2 = B2.test(Negative_ML, single_key = False)
        FP_tot = sum(FP_B2) + len(FP_ML)
        print('Threshold: %f, False positive items: %d' %(round(threshold, 2), FP_tot))
        if FP_opt > FP_tot:
            FP_opt = FP_tot
            optimal_threshold = threshold
            optimal_B1 = B1
            optimal_B2 = B2

    return optimal_B1, optimal_B2, optimal_threshold

if __name__ == '__main__':
    print(len(positive_sample))
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    optimal_B1, optimal_B2, thres_opt = Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample)

    '''Stage 2: Run LBF on all the samples'''
    ### Test queries
    B1FalsePositive = optimal_B1.test(negative_sample['url'], single_key = False)
    FP_ML = negative_sample['url'][(B1FalsePositive > thres_opt) & (negative_sample['score'] > thres_opt)]
    Negative_ML = negative_sample['url'][(B1FalsePositive > thres_opt) & (negative_sample['score'] <= thres_opt)]
    BF_positive = optimal_B2.test(Negative_ML, single_key = False)
    FP_items = sum(BF_positive) + len(FP_ML)
    FPR = FP_items/len(negative_sample)
    print('False positive items: {}; FPR: {}; Size of queries: {}'.format(FP_items, FPR, len(negative_sample)))

