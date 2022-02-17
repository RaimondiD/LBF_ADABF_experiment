from operator import pos
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import math
import serialize
import pickle
from Bloom_filter import BloomFilter
from abstract_filter import Abstract_Filter

class SLBF(Abstract_Filter):
    def __init__(self, keys, filter_size_b1, filter_size_b2, threshold):
        '''
        keys: array nella forma
            data    score
        '''
        self.filter_size_b1 = filter_size_b1
        self.filter_size_b2 = filter_size_b2
        self.threshold = threshold

        self.initial_keys = keys
        self.initial_bf = BloomFilter(len(self.initial_keys), filter_size_b1 * len(self.initial_keys)) #salvare len prima
        self.initial_bf.insert(self.initial_keys.iloc[:, 1])
        self.backup_keys = keys[(keys.iloc[:, -1] <= threshold)]
        self.backup_bf = BloomFilter(len(self.backup_keys), filter_size_b2 * len(self.initial_keys))
        self.backup_bf.insert(self.backup_keys.iloc[:, 1])

    def query(self, query_set):
        '''
        query_set: array nella forma
            data    score
        '''

        ml_false_positive = (query_set.iloc[:, -1] > self.threshold) # maschera falsi positivi generati dal modello rispetto alla soglia considerata,
        ml_true_negative = (query_set.iloc[:, -1] <= self.threshold) # maschera veri negativi generati dal modello rispetto alla soglia considerata
        # Calcolo FPR
        initial_bf_false_positive = self.initial_bf.test(query_set.iloc[:, 1], single_key = False)
        ml_false_positive_list = query_set.iloc[:, 1][(initial_bf_false_positive) & (ml_false_positive)]
        ml_true_negative_list = query_set.iloc[:, 1][(initial_bf_false_positive) & (ml_true_negative)]
        backup_bf_false_positive = self.backup_bf.test(ml_true_negative_list, single_key = False)
        total_false_positive = sum(backup_bf_false_positive) + len(ml_false_positive_list)

        return total_false_positive


def train_slbf(filter_size, query_train_set, keys, quantile_order):
    train_dataset = np.array(pd.concat([query_train_set, keys]).iloc[:, -1])
    thresholds_list = [np.quantile(train_dataset, i * (1 / quantile_order)) for i in range(1, quantile_order)] if quantile_order < len(train_dataset) else np.sort(train_dataset)
    thresh_third_quart_idx = (3 * len(thresholds_list) - 1) // 4

    fp_opt = query_train_set.shape[0]
    slbf_opt = None #cambiare
    
    for threshold in thresholds_list:
        ml_false_positive = (query_train_set.iloc[:, -1] > threshold) # maschera falsi positivi generati dal modello rispetto alla soglia considerata,
        ml_false_negative = (keys.iloc[:, -1] <= threshold) # maschera falsi negativi generati dal modello rispetto alla soglia considerata

        FP = (query_train_set[ml_false_positive].iloc[:, 1].size) / query_train_set.iloc[:, 1].size # stima probabilità di un falso positivo dal modello
        FN = (keys[ml_false_negative].iloc[:, 1].size) / keys.iloc[:, 1].size # stima probabilità di un falso negativo dal modello

        if (FP == 0.0 or FP == 1.0) or (FN == 1.0 or FN == 0.0): continue

        b2 = FN * math.log(FP / ((1 - FP) * ((1/FN) - 1)), 0.6185)
        b1 = filter_size - b2
        if b1 <= 0: # Non serve avere SLBF
            print("b1 = 0")
            break

        print(f"FP: {FP}, FN: {FN}, b: {filter_size}, b1: {b1}, b2: {b2}")

        slbf = SLBF(keys, b1, b2, threshold)
        fp_items = slbf.query(query_train_set)
        print(f"Soglia attuale: {threshold}, FP_items: {fp_items}")
        if fp_items < fp_opt:
            fp_opt = fp_items
            slbf_opt = slbf
    
    return slbf_opt, fp_opt
    
def load_filter(path):
    with open(path,"rb") as filter_file:
        slbf = pickle.load(filter_file)
    return slbf

def main(DATA_PATH_train, R_sum, others):
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
    slbf_opt, fp_opt = train_slbf(b, train_negative_sample, positive_sample, thresholds_q)

    return slbf_opt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = "store", dest = "data_path", type = str, required = True, help = "path of the dataset")
    parser.add_argument('--size_of_Sandwiched', action = "store", dest = "R_sum", type = int, required = True, help = "size of the Ada-BF")
    result =parser.parse_known_args()  
    main(result[0].data_path, result[0].R_sum, result[1])


