from os import path
import pandas as pd
import json
import argparse
from pandas.core.frame import DataFrame
from sklearn.svm import LinearSVC
from scipy.special import expit
import serialize
from numpy.core.fromnumeric import _clip_dispatcher

config_path = "models/classifier_conf.json"
path_classifier = "model/"
path_score = "score_classifier/"
classifier_fun = {"SVM" : lambda: my_svm,          #dizionario, associa ad ogni chiave la funzione associata
                   "RF" : lambda: my_rf}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                                               


parser = argparse.ArgumentParser()
parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
parser.add_argument("--dataset", action = "store", dest = "dataset_path", type = str, required = True, help = "path of the used dataset" )
result = parser.parse_args()
     # e train



def train_classifier(X_train, y_train, url, feature_vector):
    model_list, path_score_list, path_model_list = get_classifiers()
    for model, path_score, path_model  in zip(model_list, path_score_list, path_model_list):  
        print(model.__dict__)
        print (X_train)
        print(y_train)
        model.fit(X_train,y_train)
        serialize.save_score(model,feature_vector,url,path_score)
        serialize.save_model(model,path_model)


        
def get_classifiers():   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''

    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = result.classifier_list
        cl_dict = {key : data[key] for key in cl_list}
    return get_name_and_classifier(cl_dict)

def get_name_and_classifier(cl_dict):
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    path_score_list = [path_score + key for key in cl_dict]
    path_model_list = [path_classifier + key for key in cl_dict]
    return train_list, path_score_list, path_model_list

    
def my_svm(C=1):

    class My_SVM(LinearSVC):
        ''' re-implementazione delle linear SVM fornendo  get_probs, save_score, save_model (da inserire in ogni modello)'''
        
        C=2
        def get_probs(self,X):
            coef = self.coef_
            intercept = self.intercept_
            probs = []
            somma = 0
            for el in X:
                for cord,mul in zip(el, coef):
                    somma += cord * mul
                somma += intercept
                probs.append(expit(somma)) 
                somma = 0

    return My_SVM


def my_rf():
    class My_RF:
        def save_score(): pass
        def save_model():pass
        pass

def get_bloom_dataset():
    dataset = serialize.load_dataset(result.dataset_path)
    features = [el for el in dataset.columns if el!= 'url' and el != 'score']
    codif = dataset[features]
    X_train = codif[codif['label'] == 1].iloc[:,1:-1].to_numpy() 
    y_train = codif[codif['label']==1].iloc[:,-1].to_numpy()
    url = dataset['url']
    feature_vector = codif.iloc[:1:-1].to_numpy()
    return X_train, y_train, url, feature_vector


if __name__ == "__main__":
    train_classifier(*get_bloom_dataset())


    