from bdb import effective
from matplotlib.pyplot import axis
import numpy as np
import argparse
import pickle
import tensorflow as tf
import json
import serialize
import time
import math
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score
from numpy import ndarray
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from pathlib import Path


config_path = Path("models/classifier_conf.json")
params_path = Path("models/params_grid_search.json")
classifier_fun = {"SVM" : lambda: My_SVM,          #dizionario, associa ad ogni chiave la funzione associata
                   "RF" : lambda: My_Random_Forest,          #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                   "FFNN": lambda: MyKerasClassifier}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
metrics_dict = {"ROC" : roc_auc_score, 
        "average_precision_score" : average_precision_score,
        "f1-score" : f1_score,
        "accuracy" : accuracy_score}
#def take_Multi_Layer(epochs = 5, learning_rate = 1e-3, hidden_layer_size = 20):
#    return KerasClassifier(build_fn = get_MultiLayerPerceprton, _epochs= epochs, learning_rate = learning_rate, hidden_layer_size = hidden_layer_size)


def get_MultiLayerPerceprton(_epochs = 5, learning_rate = 1e-3 , hidden_layers_size = 20, _batch_size = None):
    return MultiLayerPerceptron(epochs = _epochs, learning_rate= learning_rate, hidden_layers_size= hidden_layers_size, batch_size = _batch_size)


class MyKerasClassifier(KerasClassifier):
    def __init__(self, build_fn=get_MultiLayerPerceprton, **sk_params):
        self.build_fn = build_fn
        super().__init__(build_fn=build_fn, **sk_params)

    def save_model(self,path):
        weights = self.model.get_weights()
        self.model.save_model(path)
    
    def load_model(self,path, data_train_test, data_test_test):
        weights = serialize.load_model(path)
        self.fit(data_train_test,data_test_test, verbose = 0)
        self.model.set_weights(weights)
        return self

class Sklean_classifier:

    def save_model(self, path):
        serialize.save_model(self,path)

    def load_model(self, path):
        return serialize.load_model(path)


class My_Random_Forest(RandomForestClassifier, Sklean_classifier):
   
    def get_params(self,deep=True):
        true_params = RandomForestClassifier().get_params(deep)
        my_params = self.__dict__
        return {key : item for key,item in my_params.items() if key in true_params}
    def predict_proba(self, X):
        #return [el[1] for el in super().predict_proba(X)]
        return super().predict_proba(X)

    def set_params(self, **params):
        for key ,el in params.items():
            self.__dict__[key] = el
        return self


class My_SVM(LinearSVC,Sklean_classifier):
    ''' re-implementazione delle linear SVM fornendo  get_probs, save_score, save_model (da inserire in ogni modello)'''
   
    def get_params(self, deep=True):
        return self.__dict__
    
  
    def predict_proba(self,X):
        coef = self.coef_
        intercept = self.intercept_        
        probs = expit(np.dot(X, coef[0]) + intercept[0])
        return probs
    
   

def flat(lista):
    ''' nel caso la lista contenga un unico elemento torni quello. il controllo viene fatto ricorsivamente. l'idea è che se ho
    [[[4,3]]] voglio ritornare [4,3]. Se ho [2] voglio tornare 2.'''
    if (type(lista) == list or type(lista) == ndarray) and len(lista) == 1:
        return flat(lista[0])
    return lista
                

class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, epochs = 5 , learning_rate = 1e-3, hidden_layers_size = 20, batch_size = None):
        super().__init__()
        # Parametri da confbest_score
        self.epochs = epochs
        self.learning_rate =  learning_rate
        self.hidden_layer_size = hidden_layers_size
        self.batch_size = batch_size
        # Struttura della rete
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer_size, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
        # Compile
        self.compile(
            optimizer = tf.optimizers.Adam(self.learning_rate), 
            loss = "binary_crossentropy",
            metrics = [
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                ]
        
        )

    def fit(self, x, y):
        super().fit(x, y, epochs = self.epochs, batch_size = self.batch_size)

    def predict_proba(self, X):
        scores = self.predict(X)

        return scores.flatten()

    def train_step(self, data):
        X, y = data # data dipende da ciò che viene passato a fit()

        with tf.GradientTape() as tape:
            y_hat = self(X, training = True)
            loss = self.compiled_loss(y, y_hat)

        grads = tape.gradient(loss, self.trainable_weights) # dloss_dweights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) # aggiornamento pesi
        self.compiled_metrics.update_state(y, y_hat) # aggiornamento metriche

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        out_dense_1 = self.dense1(inputs)
        out_dense_2 = self.dense2(out_dense_1)
        return out_dense_2
    
    
def integrate_train(dataset_train, dataset_test_filter, classifier_list, force_train, n_fold_CV, pos_ratio_clc, neg_ratio_clc, id, rs):  #metodo per capire se è necessario effettuare l'addestramento dei classificatori specificati
    _, m_list, _ = serialize.get_score_model_path(classifier_list,id)
    if (force_train):
        analysis_and_train(classifier_list,dataset_train,dataset_test_filter,n_fold_CV, pos_ratio_clc, neg_ratio_clc,id,rs)
    else:
        train_list = []
        for cl, m in zip(classifier_list,m_list):
            try:
                serialize.load_model(m)
            except:
                train_list.append(cl)
        if(len(train_list)):
            analysis_and_train(train_list, dataset_train, dataset_test_filter, n_fold_CV, pos_ratio_clc, neg_ratio_clc,id,rs)
    save_score(dataset_train,dataset_test_filter, classifier_list, id)

def train_classifiers(X_train, y_train, model_list, name_list, id):
    ''' dato il dataset e gli argomenti passati da linea di comando addestra i classificatori e salva i modelli e gli score'''
    _, path_model_list, _ = serialize.get_score_model_path(name_list, id)
    for model, path_model  in zip(model_list, path_model_list):  
        model.fit(X_train, y_train)
        model.save_model(path_model)

def save_score(dataset_train_filter, dataset_test_filter, name_list, id):
    path_score_list, path_model_list, path_test_list =  serialize.get_score_model_path(name_list,id)
    models = get_classifiers(name_list)
    X, y, url = separate_data(dataset_train_filter)
    try:
        time_score = serialize.load_time(id)
    except:
        time_score = {}
    for name, model, model_path, score_path, score_test_path in zip(name_list, models, path_model_list, path_score_list, path_test_list):
        if(name == "FFNN"):
            model.load_model(model_path,X[:2],y[:2])
        else:
            model = model.load_model(model_path)
        start = time.time()
        serialize.save_score(model, X, y, url, score_path)
        end = time.time()
        time_score[name] = (end-start)/len(url)
        serialize.save_score(model,*separate_data(dataset_test_filter),score_test_path)
    serialize.save_time(id,time_score)


def get_classifiers(classifier_list):   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''
    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = classifier_list
        cl_dict = {key : data[key] for key in cl_list}
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    return train_list


def get_params_list(classifier_list):
    params_list = []
    with open(params_path,"r") as file:
        data = json.load(file)
        for el in classifier_list:
            params_dict = data[el]
            params_classifier = {}
            for key in params_dict:
                if params_dict[key][-1] == 'list':
                    params_classifier[key] = params_dict[key][:-1]
                else:
                    start, stop, num = params_dict[key]
                    if(num == "range"):
                        params_classifier[key] = list(range(start,stop+1))
                    else:
                        params_classifier[key] = list(np.logspace(start, stop, num))
            params_list.append(params_classifier)
    return params_list   

def score_cl(y_score, y_train, classifier_result, name):
    for metrics_name, metrics_fun in metrics_dict.items():
        if metrics_name not in classifier_result[name].keys(): 
            classifier_result[name][metrics_name] = []
        value = metrics_fun(y_score,y_train)
        classifier_result[name][metrics_name].append(value)

def avg_cl(classifier_result,name_list):
    for name in name_list:
        for metric in metrics_dict:
            classifier_result[name][metric] = np.average(classifier_result[name][metric])
        

def cross_validation_analisys(X,y, models, names, params_list, n_fold_CV,rs, id):
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold(n_splits=n_fold_CV,random_state = rs,shuffle=True)
    result = {}
    max_scores = {}
    classifier_result = {}
    best_estimators = {}
    for el in names: 
        result[el] = []
        max_scores[el] = 0
        classifier_result[el] = {}

    for train,test in kf.split(X,y):    
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for estimator, params, name in zip(models, params_list, names):
            best_estimator, best_score = my_Grid_search(X_train, X_test, y_train, y_test, estimator, params)
            y_score = best_estimator.predict(X_test)
            score_cl(y_score,y_test, classifier_result, name)
            result[name].append(best_score) 
            if best_score > max_scores[name]:
                max_scores[name] = best_score
                best_estimators[name] = best_estimator 
    avg_cl(classifier_result,names)
    serialize.save_classifier_analysis(DataFrame(classifier_result),id,el)
    print(DataFrame(classifier_result))
        

    return best_estimators


def my_Grid_search(X_train, X_test, y_train, y_test, estimator, parmas):
    grid_obj = GridSearchCV(estimator, param_grid = parmas, scoring = 'f1', n_jobs=-1)
    grid_obj.fit(X_train,y_train)
    return grid_obj.best_estimator_, grid_obj.score(X_test,y_test)

def separate_data(dataset):
    X = dataset.iloc[:,1:-1].to_numpy()
    y = dataset.iloc[:,-1].replace(-1,0).to_numpy()
    key = dataset.iloc[:,0]
    return X, y, key

def get_bloom_dataset(dataset_train, pos_ratio_clc, neg_ratio_clc,rs):
    dataset_train, _ = serialize.divide_dataset(dataset_train, pos_ratio_clc, neg_ratio_clc, rs)
    X_train,y_train,_ = separate_data(dataset_train)
    return X_train, y_train



def analysis_and_train(classifier_list, dataset_train_filter, dataset_test_filter, n_fold_CV, pos_ratio_clc, neg_ratio_clc, id, rs):
    X_train,y_train = get_bloom_dataset(dataset_train_filter, pos_ratio_clc, neg_ratio_clc,rs)
    models = get_classifiers(classifier_list)
    params_list = get_params_list(classifier_list)
    best_estimators = cross_validation_analisys(X_train, y_train, models, classifier_list, params_list, n_fold_CV, rs, id)    
    models_to_train = []
    for _, item in best_estimators.items():
        models_to_train.append(item)
    train_classifiers(X_train, y_train,  models_to_train, classifier_list, id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")    
    parser.add_argument("--nfoldsCV", action= "store", dest = "nfoldsCV",type=int,default = 5, help = "number of folds used in CV (default = 5)")
    parser.add_argument("--pos_ratio", action = "store", dest = "pos_ratio", type = float, default = 1)
    parser.add_argument("--neg_ratio", action = "store", dest = "neg_ratio", type = float, default = 1)
    parser.add_argument("--pos_ratio_clc", action = "store", dest = "pos_ratio_clc", type = float, default = 1)
    parser.add_argument("--neg_ratio_clc", action = "store", dest = "neg_ratio_clc", type = float, default = 1)
    args = parser.parse_args()
    seed = 22012022
    data_path = args.data_path
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    pos_ratio_clc = args.pos_ratio_clc
    neg_ratio_clc = args.neg_ratio_clc
    rs = np.random.RandomState(seed)
    id = serialize.magic_id(data_path,[seed,pos_ratio,neg_ratio,pos_ratio_clc,neg_ratio_clc])
    dataset = serialize.load_dataset(args.data_path)
    dataset_train,dataset_test_filter = serialize.divide_dataset(dataset,pos_ratio, neg_ratio, rs)
    analysis_and_train(args.classifier_list,dataset_train,dataset_test_filter,args.nfoldsCV, pos_ratio_clc, neg_ratio_clc,id,rs)




    