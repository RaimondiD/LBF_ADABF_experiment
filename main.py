from pandas.core.frame import DataFrame
import learned_Bloom_filter
import sandwiched_Bloom_filter
import Ada_BF
import argparse
import classifier
import serialize
import os
from pathlib import Path

path_score = classifier.path_score
path_classifier = classifier.path_classifier
dizionario = {"learned_Bloom_filter" : lambda : learned_Bloom_filter.main,
            "sandwiched_learned_Bloom_filter" : lambda : sandwiched_Bloom_filter.main,
            "Ada-BF" : lambda : Ada_BF.main} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--type_filter", action = "store", dest = "type_filter", type = str, required= True, help = "type of fitler to build ")
    parser.add_argument("--force_train", action = "store_true", dest = "force_train")
    parser.add_argument('--size_of_filter', action="store", dest="size_of_filter", type=int, required=True, help="size of the filter")

    args, other = parser.parse_known_args()
    data_path = Path(args.data_path)
    classifier_list = args.classifier_list
    type_filter  = args.type_filter
    size_filter = args.size_of_filter
    classifier.integrate_train(data_path,classifier_list, args.force_train)
    structure_dict = {}
    for i,cl in enumerate(classifier_list):
        classifier_score_path = serialize.get_path(path_score,serialize.get_data_name(data_path),cl).with_suffix(".csv") 
        classifier_model_path = serialize.get_path(path_classifier, serialize.get_data_name(data_path), cl).with_suffix(".pk1") 
        classifier_size = os.path.getsize(classifier_model_path) * 8 # getsize restituisce dimensione in byte
        correct_size_filter = size_filter - classifier_size
        FP_items, FPR, size_query =dizionario[type_filter]()(classifier_score_path, correct_size_filter , other)
        structure_dict[cl] = {"false_positive items" : FP_items, "FPR" : FPR , "size query" : size_query, 
        "size_struct" : size_filter }
    results = DataFrame(structure_dict)
    print(results)
    serialize.save_results(results,type_filter)




    #domande:
    #   percentuale dataset?
    #   non considerata taglia classificatore
    #   come lavoro sulla taglia dei filtri (solo 1, + di una...)
    #   costruisco 1 tipologia di filtro?



    #model selection: prendo parametri strutturali : n_features, n_split e blocco la taglia massima

    #cose da fare:
            #model selection -> parametri strutturali  con valutazione classificatore area sotto curva ROC (AUC) e area sotto curva precision-recall-> guardo sklearn
            # e area sotto recall SVM (margine iperpiano) RF -> probabilità
            #sottrazione taglia
            #sistemo soglia
            #sistemo save model
            #parametri da input -> già fatto (?)