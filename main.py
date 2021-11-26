from pandas.core.frame import DataFrame
import learned_Bloom_filter
import sandwiched_Bloom_filter
import Ada_BF
import argparse
import classifier
import serialize
path_score = classifier.path_score
dizionario = {"learned_Bloom_filter" : lambda : learned_Bloom_filter.main,
            "sandwiched_learned_Bloom_filter" : lambda : sandwiched_Bloom_filter.main,
            "Ada-BF" : lambda : Ada_BF.main} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--type_filter", action = "store", dest = "type_filter", type = str, required= True, help = "type of fitler to build ")
    parser.add_argument("--force_train", action = "store", dest = "force_train", type = bool, default= False )
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
        FP_items, FPR, size_query =dizionario[type_filter]()(classifier_score_path, size_filter , other)
        structure_dict[cl] = {"false_positive items" : FP_items, "FPR" : FPR , "size query" : size_query, 
        "size_struct" : size_filter }
    print(DataFrame(structure_dict))




    #domande:
    #   percentuale dataset?
    #   non considerata taglia classificatore
    #   come lavoro sulla taglia (solo 1, + di una...)
    #   costruisco 1 tipologia di filtro?