from os import path
import pandas as pd
import json
from numpy import ndarray
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
import serialize
config_path = "models/classifier_conf.json"
path_classifier = "models/"
path_score = "score_classifier/"
classifier_fun = {"SVM" : lambda: My_SVM,          #dizionario, associa ad ogni chiave la funzione associata
                   "RF" : lambda: RandomForestClassifier}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                                               



def train_classifier(X_train, y_train, url, feature_vector,y, args):
    model_list, path_score_list, path_model_list = get_classifiers(args)
    for model, path_score, path_model  in zip(model_list, path_score_list, path_model_list):  
        model.fit(X_train, y_train)
        serialize.save_score(model, feature_vector, y, url, path_score)
        serialize.save_model(model,path_model)


        
def get_classifiers(args):   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''

    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = args.classifier_list
        cl_dict = {key : data[key] for key in cl_list}
        data_info = args.dataset_path.split("_")[0].split("/")[1]
    return get_name_and_classifier(cl_dict, data_info)

def get_name_and_classifier(cl_dict, data_info):
    ''' inizializza gli oggetti relativi ai classificatori utilizzati in '''
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    path_score_list = [path_score + key + data_info for key in cl_dict]
    path_model_list = [path_classifier + key + data_info for key in cl_dict]
    return train_list, path_score_list, path_model_list

    


class My_SVM(LinearSVC):
    ''' re-implementazione delle linear SVM fornendo  get_probs, save_score, save_model (da inserire in ogni modello)'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_proba(self,X):
        coef = self.coef_
        intercept = self.intercept_
        probs = []
        somma = 0
        for el in X:
            for cord,mul in zip(el, flat(list(coef))):
                somma += cord * mul
            somma += flat(intercept)
            probs.append(expit(somma)) 
            somma = 0
        return probs

def flat(lista):
    ''' nel caso la lista contenga un unico elemento torni quello. il controllo viene fatto ricorsivamente. l'idea è che se ho
    [[[4,3]]] voglio ritornare [4,3]. Se ho [2] voglio tornare 2.'''
    if (type(lista) == list or type(lista) == ndarray) and len(lista) == 1:
        return flat(lista[0])
    return lista
                
    

def get_bloom_dataset(args):
    dataset = serialize.load_dataset(args.dataset_path)
    features = [el for el in dataset.columns if el!= 'url' and el != 'score']
    X = dataset[features].iloc[:,1:-1].to_numpy()
    y = dataset[features].iloc[:,-1].to_numpy()
    url = dataset['url']
    X_train, _ ,y_train ,_ = train_test_split(X,y,train_size= 0.3)
    return (X_train, y_train, url, X,y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument("--dataset", action = "store", dest = "dataset_path", type = str, required = True, help = "path of the used dataset" )
    args = parser.parse_args()
    train_classifier(*get_bloom_dataset(args),args)

        # e train



    