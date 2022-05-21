import numpy as np
import argparse
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import os, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import serialize
import time
from pandas.core.frame import DataFrame
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score, f1_score,accuracy_score, precision_score
from numpy import ndarray
from sklearn.svm import LinearSVC
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight

config_path = Path("models/classifier_conf.json")
params_path = Path("models/params_grid_search.json") #sposto serealize
classifier_fun = {"SVM" : lambda: My_SVM,          #dizionario, associa ad ogni chiave la funzione associata
                   "RF" : lambda: My_Random_Forest,          #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                   "FFNN": lambda: MyKerasClassifier}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
metrics_dict = {"accuracy" : accuracy_score,
        "precision" : precision_score,
        "recall" : recall_score,
        "f1-score" : f1_score,
        "ROC" : roc_auc_score, 
        "average_precision_score" : average_precision_score,

}
#def take_Multi_Layer(epochs = 5, learning_rate = 1e-3, hidden_layer_size = 20):
#    return KerasClassifier(build_fn = get_MultiLayerPerceprton, _epochs= epochs, learning_rate = learning_rate, hidden_layer_size = hidden_layer_size)


def get_MultiLayerPerceprton(_epochs = 5, learning_rate = 1e-3 , hidden_layers_size = 20, _batch_size = None):
    return MultiLayerPerceptron(epochs = _epochs, learning_rate= learning_rate, hidden_layers_size= hidden_layers_size, batch_size = _batch_size)

class Sklearn_classifier:

    def save_model(self, path,not_serialize = False):
        return serialize.save_model(self,path,not_serialize)

    def load_model(self, path):
        return serialize.load_model(path)
    
    def get_size(self):
        return self.save_model("a",True)


class MyKerasClassifier(KerasClassifier,Sklearn_classifier):
    def __init__(self, build_fn=get_MultiLayerPerceprton, **sk_params):
        self.build_fn = build_fn
        self.hidden_layer = sk_params['hidden_layers_size']
        super().__init__(build_fn=build_fn, **sk_params)

    def save_model(self,path,not_serialize = False):
        weights = self.model.get_weights()
        return serialize.save_model(weights,path, not_serialize)

    def load_model(self,path, data_train_test, data_test_test):
        weights = serialize.load_model(path)
        self.fit(data_train_test,data_test_test) #necessario per caricare il modello (sono in un wrapper) 
        self.model.set_weights(weights)
        return self
    def __str__(self):
        return "FFNN" + str(self.hidden_layer)



class My_Random_Forest(RandomForestClassifier, Sklearn_classifier):
   
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
    
    def __str__(self):
        return "RF" + str(self.n_estimators)



class My_SVM(LinearSVC,Sklearn_classifier):
    ''' re-implementazione delle linear SVM fornendo  get_probs, save_score, save_model (da inserire in ogni modello)'''
    def get_params(self, deep=True):
        return self.__dict__
    
  
    def predict_proba(self,X):
        coef = self.coef_
        intercept = self.intercept_        
        probs = expit(np.dot(X, coef[0]) + intercept[0])
        return probs
    
    def __str__(self) -> str:
        return "SVM"


def flat(lista):
    ''' nel caso la lista contenga un unico elemento torni quello. il controllo viene fatto ricorsivamente. l'idea è che se ho
    [[[4,3]]] voglio ritornare [4,3]. Se ho [2] voglio tornare 2.'''
    if (type(lista) == list or type(lista) == ndarray) and len(lista) == 1:
        return flat(lista[0])
    return lista
                

class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, epochs = 5 , learning_rate = 1e-3, hidden_layers_size = [20], batch_size = None):
        super().__init__()
        # Parametri da confbest_score
        self.epochs = epochs
        self.learning_rate =  learning_rate
        self.hidden_layers_size = hidden_layers_size
        self.batch_size = batch_size
        # Struttura della rete
        self.dense_layers = []
        for neurons in self.hidden_layers_size:
            self.dense_layers.append(tf.keras.layers.Dense(neurons, activation = tf.nn.relu)) 
        self.out_dense = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
        # Compile
        self.compile(
            optimizer = tf.optimizers.Adam(self.learning_rate), 
            loss = "binary_crossentropy",
            metrics = [
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                ]
        
        )

    def fit(self, x, y, sample_weight = None):
        super().fit(x, y, epochs = self.epochs, batch_size = self.batch_size, sample_weight = sample_weight, verbose=2)

    def predict_proba(self, X):
        scores = self.predict(X, batch_size=5000)

        return scores.flatten()

    def train_step(self, data):
        # data dipende da ciò che è passato a fit
        if len(data) == 3:
            X, y, sample_weight = data
        else:
            sample_weight = None
            X, y = data

        with tf.GradientTape() as tape:
            y_hat = self(X, training = True)
            loss = self.compiled_loss(y, y_hat, sample_weight = sample_weight)

        grads = tape.gradient(loss, self.trainable_weights) # dloss_dweights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) # aggiornamento pesi
        self.compiled_metrics.update_state(y, y_hat, sample_weight = sample_weight) # aggiornamento metriche

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        hidden_dense_out = self.dense_layers[0](inputs)
        for layer in self.dense_layers[1:]:
            hidden_dense_out = layer(hidden_dense_out)
        output_dense_out = self.out_dense(hidden_dense_out)

        return output_dense_out
    
    
def integrate_train(dataset_train, dataset_test_filter, classifier_list, force_train, n_fold_CV, pos_ratio_clc, neg_ratio_clc, id, rs, params,balance_classes):  #metodo per capire se è necessario effettuare l'addestramento dei classificatori specificati
    train_list = []
    update_dict(params,classifier_list)
    changes = False
    s_list, m_list, t_list = serialize.get_score_model_path(get_cl_list(get_classifiers(classifier_list)),id)
    if (force_train):
        changes = True
        analysis_and_train(classifier_list, dataset_train, n_fold_CV, pos_ratio_clc, neg_ratio_clc,id,rs, balance_classes)
    else:
        for cl, m in zip(classifier_list,m_list):
            try:
                serialize.load_model(m)
            except:
                train_list.append(cl)
        if(len(train_list)):
            changes = True
            analysis_and_train(train_list, dataset_train, n_fold_CV, pos_ratio_clc, neg_ratio_clc,id,rs,balance_classes)
    save_score(dataset_train,dataset_test_filter, classifier_list, id)
    return s_list,m_list,t_list,changes

def train_classifiers(X_train, y_train, model_list, name_list, id, balance_classes):
    ''' dato il dataset e gli argomenti passati da linea di comando addestra i classificatori e salva i modelli e gli score'''
    _, path_model_list, _ = serialize.get_score_model_path(name_list, id)
    for model, path_model  in zip(model_list, path_model_list):  
        sample_weight = compute_sample_weight(class_weight='balanced', y= y_train)
        if balance_classes:
            model.fit(X_train, y_train, sample_weight = sample_weight)
        else:
            model.fit(X_train, y_train)
        model.save_model(path_model)

def save_score(dataset_train_filter, dataset_test_filter, name_list, id):
    models = get_classifiers(name_list)
    path_score_list, path_model_list, path_test_list =  serialize.get_score_model_path(get_cl_list(models),id)
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
        if(not(dataset_test_filter.empty)):
            serialize.save_score(model,*separate_data(dataset_test_filter),score_test_path)
    serialize.save_time(id,time_score)

'''
def get_hyperparameters_confs(classifier_list):
    Legge il file classifier_conf.json e restituisce tutte le combinazioni di iperparametri necessarie per ciascuno dei modelli.
    hyperpars_list = {cl : [] for cl in classifier_list}

    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = classifier_list
        cl_dict = {key : data[key] for key in cl_list}

    for cl, cl_hyperpars in cl_dict.items(): # iterazione su tutti i dizionari degli iperparametri di ogni clc
        hyperpar_combinations = list(product(*[value for value in cl_hyperpars.values()]))
        for hp_combination in hyperpar_combinations:
            hyperpar_conf = {key : value for key, value in zip(cl_hyperpars.keys(), hp_combination)}
            hyperpars_list[cl].append(hyperpar_conf)

    return hyperpars_list
'''

def get_classifiers(classifier_list):   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''
    data = serialize.get_classifiers_params(config_path)
    cl_dict = {key : data[key] for key in classifier_list}
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    return train_list

def get_params_list(classifier_list):
    params_list = []
    data = serialize.get_classifiers_params(params_path)
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

def score_cl(y_score, y_train, classifier_result, name, size, time):
    for metrics_name, metrics_fun in metrics_dict.items():
        if metrics_name not in classifier_result[name].keys(): 
            classifier_result[name][metrics_name] = []
        value = 0.5
        try: value = metrics_fun(y_score,y_train)
        except Exception: print(f"Can't compute {metrics_fun.__name__}. The value for this metric will be set to 0.5")
        if np.isnan(value):
           print("Nan found: metto a 0")
           value = 0
        classifier_result[name][metrics_name].append(value)
    classifier_result[name]["model_size"] = size
    classifier_result[name]["avg_predtime"] = time

def avg_cl(classifier_result,name_list):
    for name in name_list:
        for metric in metrics_dict:
            classifier_result[name][metric] = np.round(np.average(classifier_result[name][metric]), 5)

def cross_validation_analisys(X,y, models, names, params_list, n_fold_CV,rs, id, balance_classes):
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold(n_splits = n_fold_CV, random_state = rs, shuffle = True)
    result = {}
    max_scores = {}
    classifiers_result = {}
    best_estimators = {}
    for el in names: 
        result[el] = []
        max_scores[el] = -1
        classifiers_result[el] = {}

    for train,test in kf.split(X,y):    
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for estimator, params, name in zip(models, params_list, names):
            best_estimator, best_score = my_Grid_search(X_train, X_test, y_train, y_test, estimator, params, nfolds=n_fold_CV,
                                                        balance_classes=balance_classes)
            t_start = time.time()
            y_score = best_estimator.predict(X_test)
            t_end = time.time()
            avg_time = (t_end - t_start)/len(X_test)
            score_cl(y_score,y_test, classifiers_result, name, best_estimator.get_size(), avg_time)
            result[name].append(best_score) 
            if best_score > max_scores[name]:
                max_scores[name] = best_score
                best_estimators[name] = best_estimator 
    avg_cl(classifiers_result,names)
    for el in names:
        classifier_result = DataFrame(classifiers_result[el],index=[el])
        serialize.save_classifier_analysis(classifier_result,id,el)
        print(classifier_result)
    total = DataFrame.from_dict(classifiers_result, orient="index")
    serialize.save_classifier_analysis(total,id,"total")

    return best_estimators

def my_Grid_search(X_train, X_test, y_train, y_test, estimator, params, nfolds, balance_classes):
    sample_weight = compute_sample_weight(class_weight='balanced', y= y_train)
#    vals = np.unique(sample_weight)
#    print(vals)
#    minv = np.min(vals)
#    maxv = np.max(vals)
#    maxw = np.maximum(minv, maxv*0.33)
#    sample_weight[sample_weight == maxv] = maxw
#    print(sample_weight)
#    print("balance_classes:", balance_classes)
    grid_obj = GridSearchCV(estimator, param_grid = params, scoring = 'f1', cv=nfolds)
    if balance_classes:
        grid_obj.fit(X_train, y_train, sample_weight = sample_weight)
    else:
        grid_obj.fit(X_train, y_train)
#    grid_obj.fit(X_train, y_train)    
    return grid_obj.best_estimator_, grid_obj.score(X_test,y_test)

def separate_data(dataset):
    X = dataset.iloc[:,1:-1].to_numpy()
    y = dataset.iloc[:,-1].replace(-1,0).to_numpy()
    key = dataset.iloc[:,0]
    return X, y, key

def get_bloom_dataset(dataset, pos_ratio_clc, neg_ratio_clc, rs, id):
    dataset_train, _ = serialize.divide_dataset(dataset, None, pos_ratio_clc, neg_ratio_clc, 0, rs)
    print(f"Samples for classifiers training + testing: {len(dataset_train.index)}. (Pos, Neg): ({len(dataset_train[(dataset_train['label'] == 1)])}, {len(dataset_train[(dataset_train['label'] == -1)])})")
    serialize.save_dataset_info(dataset, dataset_train, id)
    X_train,y_train,_ = separate_data(dataset_train)

    return X_train, y_train

def analysis_and_train(classifier_list, dataset_train_filter, n_fold_CV, pos_ratio_clc, neg_ratio_clc, id, rs, balance_classes):
    X_train,y_train = get_bloom_dataset(dataset_train_filter, pos_ratio_clc, neg_ratio_clc, rs, id)
    models = get_classifiers(classifier_list)
    params_list = get_params_list(classifier_list)
    classifier_list = get_cl_list(models)
    best_estimators = cross_validation_analisys(X_train, y_train, models, classifier_list, params_list, n_fold_CV, rs, id, balance_classes)    
    models_to_train = []
    for _, item in best_estimators.items():
        models_to_train.append(item)
    print(len(models_to_train))
    train_classifiers(X_train, y_train, models_to_train, classifier_list, id, balance_classes)
    return classifier_list

def get_cl_list(models):
    return list(map(lambda x : str(x), models))

def update_dict(params,classifier_list):
    data = serialize.get_classifiers_params(config_path)
    for el,name,cl in params:
        if el != None and cl in classifier_list:
            if type(el) == list:
                el = list(map(lambda x : int(x), el))
            else:
                el = int(el)
            data[cl][name] = el
    serialize.save_classifiers_params(data,config_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")    
    parser.add_argument("--nfoldsCV", action= "store", dest = "nfoldsCV",type=int,default = 5, help = "number of folds used in CV (default = 5)")
    parser.add_argument("--pos_ratio", action = "store", dest = "pos_ratio", type = float, default = 1)
    parser.add_argument("--neg_ratio", action = "store", dest = "neg_ratio", type = float, default = 1)
    parser.add_argument("--pos_ratio_clc", action = "store", dest = "pos_ratio_clc", type = float, default = 1)
    parser.add_argument("--neg_ratio_clc", action = "store", dest = "neg_ratio_clc", type = float, default = 1)
    parser.add_argument("--trees", action = "store", dest = "tree_param", type = str, default = None )
    parser.add_argument("--layers", action = "store", dest = "layer_size_param", type = str, nargs = '+', default = None )
    parser.add_argument("--negTest_ratio", action = "store", dest = "negTest_ratio", type = float, default = 0)
    parser.add_argument('--test_path', action = "store", dest = "test_path", type = str, default = None)  
    parser.add_argument('--balance_classes', action = "store_true", dest = "balance_classes")  
    

    args = parser.parse_args()
    seed = 22012022
    data_path = Path(args.data_path)
    params = [(args.tree_param,"n_estimators","RF"),(args.layer_size_param,"hidden_layers_size","FFNN")]

    data_test_path = Path(args.test_path) if args.test_path is not None else None
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    pos_ratio_clc = args.pos_ratio_clc
    neg_ratio_clc = args.neg_ratio_clc
    negTest_ratio = args.negTest_ratio

    rs = np.random.RandomState(seed)
    id = serialize.magic_id(data_path,[seed,pos_ratio,neg_ratio,pos_ratio_clc,neg_ratio_clc])
    dataset = serialize.load_dataset(data_path)
    dataset_train,dataset_test_filter = serialize.divide_dataset(dataset, data_test_path,
                                                    pos_ratio, neg_ratio, negTest_ratio, rs)
    del(dataset)
    gc.collect()
    update_dict(params,args.classifier_list)
    analysis_and_train(args.classifier_list,dataset_train,args.nfoldsCV, pos_ratio_clc, neg_ratio_clc,id,rs, args.balance_classes)




    
