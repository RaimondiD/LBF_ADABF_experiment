import pandas as pd
import pickle 
import os
import numpy as np
from pathlib import Path

result_path = Path("results/")

def load_dataset(path):
    return pd.read_csv(path)

def save_results(dict, filter_name):
    result_path.mkdir(parents = True, exist_ok = True)
    dict.to_csv(result_path / filter_name)

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
    save_object.to_csv(save_path.with_suffix(".csv"))
        
def get_data_name(data_path):
    '''Datapath nella forma path_to_dataset/datasetname_data.csv'''
    return data_path.parts[-1].split("_")[0]

def get_path(dir_path, data_name, classifier):
    return dir_path / data_name / classifier 
