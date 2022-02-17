from pandas.core.frame import DataFrame
import learned_Bloom_filter
import sandwiched_Bloom_filter
import Ada_BF
import argparse
import classifier
import serialize
import os
import time
import numpy as np
from pathlib import Path


path_score = serialize.path_score
path_score_test = serialize.path_score_test
path_classifier = serialize.path_classifier
dizionario = {"learned_Bloom_filter" : lambda : learned_Bloom_filter.main,
            "sandwiched_learned_Bloom_filter" : lambda : sandwiched_Bloom_filter.main,
            "Ada-BF" : lambda : Ada_BF.main} 

if __name__ == "__main__":
    seed= 110011
    rs = np.random.RandomState(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--type_filter", action = "store", dest = "type_filter", type = str, required= True, help = "type of fitler to build ")
    parser.add_argument("--force_train", action = "store_true", dest = "force_train")
    parser.add_argument('--size_of_filter', action="store", dest="size_of_filter", type=int, required=True, help="size of the filter")
    parser.add_argument("--nfoldsCV", action= "store", dest = "nfoldsCV", type=int, default = 5, help = "number of fold used in CV (default = 5)")
    parser.add_argument("--pos_ratio", action = "store", dest = "pos_ratio", type = float, default = 0.7)
    parser.add_argument("--neg_ratio", action = "store", dest = "neg_ratio", type = float, default = 0.7)
    parser.add_argument("--pos_ratio_clc", action = "store", dest = "pos_ratio_clc", type = float, default = 0.7)
    parser.add_argument("--neg_ratio_clc", action = "store", dest = "neg_ratio_clc", type = float, default = 0.7)
    parser.add_argument("--negTest_ratio", action = "store", dest = "negTest_ratio", type = float, default = 1.0)

    args, other = parser.parse_known_args()
    data_path = Path(args.data_path)
    classifier_list = args.classifier_list
    type_filter  = args.type_filter
    size_filter = args.size_of_filter
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    negTest_ratio = args.negTest_ratio
    pos_ratio_clc = args.pos_ratio_clc
    neg_ratio_clc = args.neg_ratio_clc
    if(pos_ratio >= 1 or neg_ratio >= 1 or pos_ratio <=0 or neg_ratio <=0 ):
        raise AssertionError("pos_ration and neg_ratio must be > 0 and < 1 ")

    '''
    Suddivisione del dataset
    '''
    dataset = serialize.load_dataset(data_path, dtype = np.int8)
    print(len(dataset.index))
    dataset_train, other_dataset = serialize.divide_dataset(dataset,pos_ratio,neg_ratio,rs)
    dataset_test_filter, _ = serialize.divide_dataset(other_dataset, 0, negTest_ratio, rs)
    print(len(dataset_train.index), len(dataset_test_filter.index))
    id = serialize.magic_id(data_path,[seed, pos_ratio, neg_ratio, pos_ratio_clc, neg_ratio_clc])
    '''
    Addestramento e salvataggio dei classificatori
    '''
    classifier.integrate_train(dataset_train, dataset_test_filter, classifier_list, args.force_train, args.nfoldsCV, pos_ratio_clc, neg_ratio_clc, id, rs)
    structure_dict = {}
    cl_time = serialize.load_time(id)
    '''
    Creazione e test dei filtri
    '''
    for i,cl in enumerate(classifier_list):
        ### Generazione path con score e modelli
        classifier_score_path = serialize.get_path(path_score,Path(id),cl).with_suffix(".csv") 
        classifier_model_path = serialize.get_path(path_classifier, Path(id), cl).with_suffix(".pk1") 
        classifier_score_path_test = serialize.get_path(path_score_test,Path(id),cl).with_suffix(".csv") 
        classifier_size = os.path.getsize(classifier_model_path)
        correct_size_filter = size_filter - classifier_size
        if correct_size_filter < 0:
            print(f"size of classifier {cl} is greater than budget")
            continue
        ### Creazione filtro
        filter_opt = dizionario[type_filter]()(classifier_score_path, correct_size_filter, other)
        ### Query di test
        negative_sample_test = serialize.load_dataset(classifier_score_path_test)
        start = time.time()
        fp_items = filter_opt.query(negative_sample_test)
        end = time.time()
        ### Salvataggio risultati
        fpr = fp_items/len(negative_sample_test)
        filter_time = (end-start)/len(negative_sample_test)
        structure_dict[cl] = {"FPR" : fpr , "size_struct" : size_filter, "size_classifier" : classifier_size , "medium query time: ": cl_time[cl] + filter_time}
    if len(structure_dict) ==0:
        print(f"budget is too low to create a {type_filter}")
    else : 
        results = DataFrame(structure_dict)
        print(results)
        serialize.save_results(results,type_filter)



