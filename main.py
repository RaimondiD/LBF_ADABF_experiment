import learned_Bloom_filter
#import sandwiched_Bloom_filter
#import Ada_BF
import argparse
import classifier
import serialize

dizionario = {"LBF" : lambda : main}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--type_filter", action = "store", dest = "type_filter", type = str, required= True, help = "type of fitler to build ")
    parser.add_argument("--force_train", action = "store", dest = "force_train", type = bool, default= False )
    parser.add_argument('--size_of_filter', action="store", dest="R_sum", type=int, required=True,
                    help="size of the filter")

    args = parser.parse_args()
    data_path = args.data_path
    classifier_list = args.classifier_list
    type_filter  = args.type_filter
    classifier.integrate_train(data_path,classifier_list, args.force_train)
