from genericpath import exists
import pandas as pd
import pickle 
import os
import numpy as np
from pathlib import Path
result_path = Path("results/")
path_classifier = Path("models/")
path_score = Path("score_classifier/")

def load_dataset(path):
    return pd.read_csv(path)

def load_time(data_path):
    total_path = get_time_path(data_path)
    return load_model(total_path)

def save_time(data_path,model):
    total_path = get_time_path(data_path)
    save_model(model,total_path)
    
def get_time_path(data_path):
    data_info = get_data_name(data_path)
    total_path = get_path(path_classifier,data_info,"time")
    dest_dir = path_classifier / data_info
    dest_dir.mkdir(parents= True, exist_ok = True)
    return total_path

def save_results(dict, filter_name):
    result_path.mkdir(parents = True, exist_ok = True)
    dict.to_csv(result_path / filter_name)

def load_model(path):
    path = get_model_path(path)
    with open(path,"rb") as model_file:
        model = pickle.load(model_file)
    return model


def save_classifier_analysis(dict,data_path,classifier):
    data_name = get_data_name(data_path)
    dest_dir = result_path / data_name 
    dest_dir.mkdir(parents= True, exist_ok = True)
    dict.to_csv(dest_dir / Path(classifier + "_score"))

def save_model(model, save_path):
    '''salva i modelli'''
    save_path = get_model_path(save_path)
    with open(save_path,'wb') as file:
            pickle.dump(model,file)
    size = os.path.getsize(save_path)
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
    d = {'url' : url, 'label' : y, 'score' : score}
    save_object = pd.DataFrame(d)
    save_object.to_csv(save_path)
        
def get_data_name(data_path):
    '''Datapath nella forma path_to_dataset/datasetname_data.csv'''
    data_path = Path(data_path)
    return data_path.parts[-1].split("_")[0]

def get_path(dir_path, data_name, classifier):
    return dir_path / data_name / classifier 


def get_score_model_path(cl_dict, data_path):
    data_info = get_data_name(data_path)
    ps = Path(path_score / data_info)
    pc = Path(path_classifier / data_info)
    ps.mkdir(parents = True, exist_ok = True)
    pc.mkdir(parents = True, exist_ok = True)
    # serialize.try_to_solve(path_score + data_info)
    # serialize.try_to_solve(path_classifier + data_info)
    path_score_list = [get_path(path_score, data_info, key).with_suffix(".csv") for key in cl_dict]
    path_model_list = [get_path(path_classifier, data_info, key).with_suffix(".pk1") for key in cl_dict]
    return path_score_list, path_model_list