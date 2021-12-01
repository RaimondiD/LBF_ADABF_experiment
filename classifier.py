from os import path
import os
from pandas.core.frame import DataFrame
import tensorflow as tf
import json
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy import ndarray
import numpy as np
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import serialize

config_path = "models/classifier_conf.json"
params_path = "models/params_grid_search.json"
path_classifier = "models/"
path_score = "score_classifier/"
classifier_fun = {"SVM" : lambda: My_SVM,          #dizionario, associa ad ogni chiave la funzione associata
                   "RF" : lambda: My_Random_Forest,          #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                   "FFNN": lambda: take_Multi_Layer}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                                               
def take_Multi_Layer(epochs = 5, learning_rate = 1e-3, hidden_layer_size = 20):
    return KerasClassifier(build_fn = get_MultiLayerPerceprton, _epochs= epochs, learning_rate = learning_rate, hidden_layer_size = hidden_layer_size)


def get_MultiLayerPerceprton(_epochs = 5, learning_rate = 1e-3 , hidden_layer_size = 20):
    return MultiLayerPerceptron(epochs = _epochs, learning_rate= learning_rate, hidden_layers_size= hidden_layer_size)
    

class My_Random_Forest(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
    def save_model(self, path):
        serialize.save_model(self,path)

class My_SVM(LinearSVC):
    ''' re-implementazione delle linear SVM fornendo  get_probs, save_score, save_model (da inserire in ogni modello)'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return self.__dict__
    
    def save_model(self,path):
        serialize.save_model(self,path)

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
                

class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, epochs = 5 , learning_rate = 1e-3, hidden_layers_size = 20):
        super().__init__()
        # Parametri da conf
        self.epochs = epochs
        self.learning_rate =  learning_rate
        self.hidden_layer_size = hidden_layers_size
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
        print(self.epochs)
        super().fit(x, y, epochs = self.epochs)

    def predict_proba(self, X):
        scores = self.predict(X)

        return scores.flatten().tolist()

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
    



def integrate_train(data_path, classifier_list, force_train):  #metodo per capire se è necessario effettuare l'addestramento dei classificatori specificati
    if (force_train):
        analysis_and_train(classifier_list, data_path)
    else:
        train_list = []
        for cl in classifier_list:
            path_s = serialize.get_path(path_score, serialize.get_data_name(data_path), cl) + ".csv" #ricava il path a cui dovrebbero essere salvati gli score, nel caso siano stati calcolati
            #path_m = serialize.get_path(path_classifier, serialize.get_data_name(data_path), cl) + ".pk1"
            try:
                open(path_s) #and open(path_m)
            except:
                train_list.append(cl)
        if(len(train_list)):
            analysis_and_train(train_list, data_path)

def train_classifiers(X_train, y_train, url, X, y, model_list, path_score_list, path_model_list):
    ''' dato il dataset e gli argomenti passati da linea di comando addestra i classificatori e salva i modelli e gli score'''
    for model, path_score, path_model  in zip(model_list, path_score_list, path_model_list):  
        model.fit(X_train, y_train)
        serialize.save_score(model, X, y, url, path_score)
        try:
            serialize.save_model(model,path_model)
        except: pass

def get_classifiers(classifier_list, data_path):   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''
    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = classifier_list
        cl_dict = {key : data[key] for key in cl_list}
        data_info = os.path.split(data_path.split("_")[0])[1]
    return get_path_and_classifier(cl_dict, data_info)

def get_path_and_classifier(cl_dict, data_info):
    ''' inizializza gli oggetti relativi ai classificatori utilizzati in '''
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    path_score_list = [serialize.get_path(path_score, data_info, key) for key in cl_dict]
    path_model_list = [serialize.get_path(path_classifier, data_info, key) for key in cl_dict]
    return train_list, path_score_list, path_model_list

def get_params_list(classifier_list):
    params_list = []
    with open(params_path,"r") as file:
        data = json.load(file)
        for el in classifier_list:
            params_dict = data[el]
            params_classifier = {}
            for key in params_dict:
                start, stop, num = params_dict[key]
                if(num == "range"):
                    params_classifier[key] = list(range(start,stop+1))
                else:
                    params_classifier[key] = np.logspace(start, stop, num)
            params_list.append(params_classifier)
    return params_list   


def cross_validation_analisys(X,y, models, names, params_list):
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold()
    result = {}
    max_scores = {}
    best_estimators = {}
    for el in names: 
        result[el] = []
        max_scores[el] = 0
    for train,test in kf.split(X,y):    
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for estimator, params, name in zip(models, params_list, names):
            best_estimator, best_score = my_Grid_search(X_train, X_test, y_train, y_test, estimator, params)
            result[name].append(best_score) 
            if best_score > max_scores[name]:
                max_scores[name] = best_score
                best_estimators[name] = best_estimator    
    print(DataFrame(result))
    return best_estimators


def my_Grid_search(X_train, X_test, y_train, y_test, estimator, parmas):
    grid_obj = GridSearchCV(estimator, param_grid = parmas, scoring = 'f1', cv = 2)
    grid_obj.fit(X_train,y_train)
    return    grid_obj.best_estimator_, grid_obj.score(X_test,y_test)


def get_bloom_dataset(data_path):
    dataset = serialize.load_dataset(data_path)
    features = [el for el in dataset.columns if el!= 'url' and el != 'score']
    X = dataset[features].iloc[:,1:-1].to_numpy()
    y = dataset[features].iloc[:,-1].replace(-1, 0).to_numpy() # .replace(-1, 0) per binary loss
    url = dataset['url']
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 0.7, random_state=42)
    return X_train, y_train, X_test, y_test, url, X,y

def analysis_and_train(classifier_list, data_path):
    X_train, y_train, X_test, y_test, url, X, y = get_bloom_dataset(data_path)
    models,path_score_list ,path_model_list = get_classifiers(classifier_list, data_path)
    params_list = get_params_list(classifier_list)
    best_estimators = cross_validation_analisys(X_train, y_train, models, classifier_list, params_list)
    for el, item in best_estimators.items():
        y_score = item.predict(X_test)
        print(f"{el} roc auc score : {roc_auc_score(y_test,y_score)}")
        print(f"{el} average precision score : {average_precision_score(y_test,y_score)}")
    train_classifiers(X_train, y_train,url, X, y, models, path_score_list, path_model_list )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")    
    args = parser.parse_args()
    analysis_and_train(args.classifier_list, args.data_path)

        # e train



    