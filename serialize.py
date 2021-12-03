from numpy.core.fromnumeric import size
import pandas as pd
import pickle 
import os
import numpy as np

result_path = "results/"

def load_dataset(path):
    return pd.read_csv(path)

def save_results(dict, filter_name):
    try_to_solve(result_path)
    dict.to_csv(result_path + filter_name)

def save_model(model, save_path):
    '''salva i modelli'''
    with open(save_path,'wb') as file:
            pickle.dump(model,file)
    size = os.path.getsize(save_path)
    return size

def save_score(model, X_test,y, url, save_path):
    '''salva i punteggi in un file csv, in modo da poterli poi utilizzare per la creazione di filtri'''         
    score = np.array(model.predict_proba(X_test))
    if len(score.shape) > 1:
        score = [el[1] for el in score]
    d = {'url' : url, 'label' : y, 'score' : score}
    save_object = pd.DataFrame(d)
    save_object.to_csv(save_path+".csv")
        

def get_data_name(data_path):
    return os.path.split(data_path.split("_")[0])[1]

def get_path(dir_path, data_name, classifier):
    return dir_path + data_name + "/"  + classifier 

def try_to_solve(path):
    try:
        os.mkdir(path)
    except:
        pass
