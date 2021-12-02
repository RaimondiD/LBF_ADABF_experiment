from pandas.core.frame import DataFrame
import learned_Bloom_filter
import sandwiched_Bloom_filter
import Ada_BF
import argparse
import classifier
import serialize
import os

path_score = classifier.path_score
path_classifier = classifier.path_classifier
dizionario = {"learned_Bloom_filter" : lambda : learned_Bloom_filter.main,
            "sandwiched_learned_Bloom_filter" : lambda : sandwiched_Bloom_filter.main,
            "Ada-BF" : lambda : Ada_BF.main} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--type_filter", action = "store", dest = "type_filter", type = str, required= True, help = "type of fitler to build ")
    parser.add_argument("--force_train", action = "store_true", dest = "force_train")
    parser.add_argument('--size_of_filter', action="store", dest="size_of_filter", type=int, required=True,
                    help="size of the filter")

    args, other = parser.parse_known_args()
    data_path = args.data_path
    classifier_list = args.classifier_list
    type_filter  = args.type_filter
    size_filter = args.size_of_filter
    classifier.integrate_train(data_path,classifier_list, args.force_train)
    structure_dict = {}

    for i,cl in enumerate(classifier_list):
        classifier_score_path = serialize.get_path(path_score,serialize.get_data_name(data_path),cl) + ".csv" 
        classifier_model_path = serialize.get_path(path_classifier, serialize.get_data_name(data_path), cl) + ".pk1" 
        classifier_size = os.path.getsize(classifier_model_path) * 8 # getsize restituisce dimensione in byte
        correct_size_filter = size_filter - classifier_size
        if correct_size_filter < 0:
            print(f"size of classifier {cl} is greater than budget")
            continue
        _, FPR, _ =dizionario[type_filter]()(classifier_score_path, correct_size_filter , other)
        structure_dict[cl] = {"FPR" : FPR , "size_struct" : size_filter, "size_classifier" : classifier_size }
    if len(structure_dict) ==0:
        print(f"budget is too low to create a {type_filter}")
    else : 
        results = DataFrame(structure_dict)
        print(results)
        serialize.save_results(results,type_filter)

