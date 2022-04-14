from sqlite3 import DataError
import pandas as pd
import pickle 
import os
import numpy as np
import lzma
import json
from pathlib import Path
import json
MAX_LEN_NAME = 24
MAX_LEN_CLASS = 3
result_path = Path("results/")
path_classifier = Path("models/")
path_score = Path("score_classifier/")
path_score_test = Path("score_classifier_test/")
magic_name_list = ["s","pr","nr","prc","nrc"] #string used to create magic_id


def divide_dataset(dataset, dataset_test, pos_ratio, neg_ratio, negTest_ratio, rs, pos_label = 1):
    neg_label = find_neg_label(dataset)
    negative = dataset.loc[(dataset['label'] == neg_label)]
    positive = dataset.loc[(dataset['label'] == pos_label)]
    del(dataset)
    neg_len, pos_len = len(negative), len(positive)
    
    # Creation of index for the training dataset of classifiers and filters
    negative_samples_train_idx = rs.choice(range(neg_len), replace = False, size = int(neg_len * neg_ratio)) # random state?
    positive_samples_train_idx = rs.choice(range(pos_len), replace = False, size = int(pos_len * pos_ratio))
    # Training dataset splitting
    negative_samples_train = negative.iloc[negative_samples_train_idx, :]
    positive_samples_train = positive.iloc[positive_samples_train_idx, :]
    del(positive)

    if dataset_test is None: # If dataset_test is None, filters testing index will be extracted from the unused part of dataset
        negative_other_idx = np.setdiff1d(np.arange(0, neg_len), negative_samples_train_idx) 
        other_negative = negative.iloc[negative_other_idx, :]
        negTest_len = len(other_negative)
        del(negative)

        # Creation of index for filters testing
        negative_samples_test_idx = rs.choice(range(negTest_len), replace = False, size = int(negTest_len * negTest_ratio))
        # Testing dataset splitting
        negative_samples_test = other_negative.iloc[negative_samples_test_idx, :]
        del(other_negative)
    else: # If dataset_test is not None, filters testing index will be extracted from dataset_test
        del(negative)
        negative_test = dataset_test.loc[(dataset_test['label'] == neg_label)]
        del(dataset_test)
        negTest_len = len(negative_test)

        # Creation of index for filters testing
        negative_samples_test_idx = rs.choice(range(negTest_len), replace = False, size = int(negTest_len * negTest_ratio))
        # Testing dataset splitting
        negative_samples_test = negative_test.iloc[negative_samples_test_idx, :]

    # Concatenazione
    train = pd.concat([negative_samples_train, positive_samples_train], axis = 0, ignore_index = True)
    del(negative_samples_train)
    del(positive_samples_train)
    train = train.sample(frac = 1, random_state= rs).reset_index(drop=True) # utile o per qualche motivo la cv si rompe con la ffnn, occhio con dataset grandi

    return train, negative_samples_test

def magic_id(data_path,list):
    result = get_data_name(data_path)
    for name,el in zip(magic_name_list,list):
        result += f"_{name}={str(el)}"
    return result

def load_dataset(path):
    json_dtypes_file = path.parent / (path.stem + "_dtypes.json")
    if(json_dtypes_file.exists()):
        with json_dtypes_file.open() as f: 
            dtypes = json.load(f)
            dtypes = {col_name : np.dtype(t) for col_name, t in dtypes.items()}
            data = pd.read_csv(path, dtype = dtypes)
    else: 
        data = pd.read_csv(path)
        
    return data

def load_time(data_path):
    total_path = get_time_path(data_path)
    return load_model(total_path)

def save_time(data_path,model):
    total_path = get_time_path(data_path)
    save_model(model,total_path)
    
def get_time_path(id):
    data_info = Path(id)
    total_path = get_path(path_classifier,data_info,"time")
    dest_dir = path_classifier / data_info
    dest_dir.mkdir(parents= True, exist_ok = True)
    return total_path

def save_results(dict, filter_name,id, test_file):
    if test_file == None:
        filter_result_path = result_path / Path(id)
        filter_result_path.mkdir(parents = True, exist_ok = True)
        dict.to_csv(filter_result_path / (filter_name + ".csv"))
    else:
        save_text_result(dict,filter_name,id, test_file)

def save_text_result(dict,filter_name,id, test_file):
    space_name = (MAX_LEN_NAME-(len(id)//4))*"\t"
    result_str = ""
    if not os.path.exists(test_file):
        result_str = "data"+MAX_LEN_NAME*"\t"+"type\tmethod"+ MAX_LEN_CLASS*"\t"+"FPR\tSPACE\n"
    type_dict = {"learned_Bloom_filter" : "LBF", "sandwiched_learned_Bloom_filter": "SLBF", "Ada-BF":"ADA-BF"}
    for key in dict:
        space_method = "\t"*(MAX_LEN_CLASS-len(key)//4)
        result_str+=f"""{id }{space_name}\t{type_dict[filter_name]}\t{key}{space_method}\t{dict[key]["FPR"]}\t{dict[key]["size_struct"]}\n"""
    with open(test_file,"a") as text_file:
        text_file.write(result_str)



def load_model(path):
    path = get_model_path(path)
    with lzma.open(path,"rb") as model_file:
        model = pickle.load(model_file)
    return model

def exist_model(path):
    path = get_model_path(path)
    return os.path.exists(path)



def save_classifier_analysis(dict, id, classifier, score = True):
    data_name = Path(id)
    dest_dir = result_path / data_name 
    dest_dir.mkdir(parents= True, exist_ok = True)
    if score:
        classifier += "_score"
    dict.to_csv(dest_dir / Path(classifier + ".csv"))

def save_model(model, save_path, not_serialize = False):
    '''salva i modelli'''
    save_path = get_model_path(save_path)
    with lzma.open(save_path,'wb') as file:
            pickle.dump(model,file)
    size = os.path.getsize(save_path) 
    if(not_serialize):
        os.remove(save_path)
    return size

def get_model_path(path):
    str_path = str(path)
    if str_path[-4:] != ".pk1":
        str_path += ".pk1"
    return Path(str_path)


def save_score(model, X_test,y, url, save_path):
    '''salva i punteggi in un file csv, in modo da poterli poi utilizzare per la creazione di filtri'''         
    score = np.array(model.predict_proba(X_test))
    if len(score.shape) > 1:
        score = [el[1] for el in score]
    d = {'data' : url, 'label' : y, 'score' : score}
    save_object = pd.DataFrame(d)
    save_object.to_csv(save_path)
        
def get_data_name(data_path):
    '''Datapath nella forma path_to_dataset/datasetname_data.csv'''
    data_path = Path(data_path)
    return data_path.parts[-1][:-4]
    #return data_path

def get_path(dir_path, data_name, classifier):
    return dir_path / data_name / classifier 

def get_list_path(dir_path, data_name, list,  suffix = None):
    if suffix:
        return [get_path(dir_path,data_name,key).with_suffix(suffix) for key in list]
    return [get_path(dir_path,data_name,key) for key in list]

def get_score_model_path(cl_dict, id):
    data_info = Path(id)
    ps = Path(path_score / data_info)
    pc = Path(path_classifier / data_info)
    ps_test = Path(path_score_test / data_info)
    ps.mkdir(parents = True, exist_ok = True)
    pc.mkdir(parents = True, exist_ok = True)
    ps_test.mkdir(parents = True, exist_ok = True)
    path_score_list = get_list_path(path_score, data_info, cl_dict,".csv")
    path_model_list = get_list_path(path_classifier, data_info, cl_dict,".pk1")
    ps_test_list =  get_list_path(path_score_test, data_info, cl_dict,".csv")
    return path_score_list, path_model_list, ps_test_list


def save_dataset_info(dataset,dataset_train, id, pos_label = 1):
    labels = {"pos" : lambda x, pos_label = pos_label : x.loc[(x['label'] == pos_label)],
            "neg": lambda x : x.loc[(x['label'] == find_neg_label(x))],
            }
    result = {}
    for label,fun in labels.items():
        result[label] = {}
        result[label]["total"] = len(fun(dataset))
        result[label]["classifier"] = len(fun(dataset_train))
    result = pd.DataFrame(result)
    print(result)
    save_classifier_analysis(result,id,"info_dataset",score = False)


def get_classifiers_params(path):
    with open(path,"r") as file:
        data = json.load(file)
    return data

def save_classifiers_params(cl_dict,path):
    with open(path,"w") as file:
        json.dump(cl_dict,file,indent="\t")

def find_neg_label(dataset):
    neg_label = -1
    if len(dataset.loc[(dataset['label'] == neg_label)]) == 0:
        neg_label = 0
    return neg_label
