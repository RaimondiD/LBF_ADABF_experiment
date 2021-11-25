from os import path
import pandas as pd
import tensorflow as tf
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
                   "RF" : lambda: My_Random_Forest,          #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                   "FFNN": lambda: MultiLayerPerceptron}           #da aggiungere FFNN; le funzioni devono disporre di un metodo train and save per l'addestramento e il salvataggio di score e parametri
                                               
    
class My_Random_Forest(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_proba(self, X):
        return [el[1] for el in super().predict_proba(X)]

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

def integrate_train(data_path, classifier_list, force_train):  #metodo per capire se è necessario effettuare l'addestramento dei classificatori specificati
    if (force_train):
        train_classifier(*get_bloom_dataset(data_path),classifier_list, data_path)
    else:
        train_list = []
        for cl in classifier_list:
            path_s = serialize.get_path(path_score, serialize.get_data_name(data_path), cl) + ".csv" #ricava il path a cui dovrebbero essere salvati gli score, nel caso siano stati calcolati
            path_m = serialize.get_path(path_classifier, serialize.get_data_name(data_path), cl) + ".pk1"
            try:
                open(path_s) and open(path_m)
            except:
                train_list.append(cl)
        if(len(train_list)):
            train_classifier(*get_bloom_dataset(data_path),train_list, data_path)
    

def train_classifier(X_train, y_train, url, feature_vector,y, classifier_list, data_path):
    ''' dato il dataset e gli argomenti passati da linea di comando addestra i classificatori e salva i modelli e gli score'''
    model_list, path_score_list, path_model_list = get_classifiers(classifier_list, data_path)
    for model, path_score, path_model  in zip(model_list, path_score_list, path_model_list):  
        model.fit(X_train, y_train)
        serialize.save_score(model, feature_vector, y, url, path_score)
        serialize.save_model(model,path_model)


        
def get_classifiers(classifier_list, data_path):   
    ''' carica il file di configurazione e ritorna le classi dei classificatori necessari, il path a cui vengono salvati 
        gli score e il path a cui vengono salvati i modelli '''

    with open(config_path,"r") as file:
        data = json.load(file)
        cl_list = classifier_list
        cl_dict = {key : data[key] for key in cl_list}
        data_info = data_path.split("_")[0].split("/")[1]
    return get_name_and_classifier(cl_dict, data_info)

def get_name_and_classifier(cl_dict, data_info):
    ''' inizializza gli oggetti relativi ai classificatori utilizzati in '''
    train_list =  [classifier_fun[key]()(**kwargs) for key, kwargs in cl_dict.items()] 
    path_score_list = [serialize.get_path(path_score, data_info, key) for key in cl_dict]
    path_model_list = [serialize.get_path(path_classifier, data_info, key) for key in cl_dict]
    return train_list, path_score_list, path_model_list



class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.dense1 = tf.keras.layers.Dense(kwargs.get('hidden_layers_size', 20), activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)

        self.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = tf.losses.BinaryCrossentropy())

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

def flat(lista):
    ''' nel caso la lista contenga un unico elemento torni quello. il controllo viene fatto ricorsivamente. l'idea è che se ho
    [[[4,3]]] voglio ritornare [4,3]. Se ho [2] voglio tornare 2.'''
    if (type(lista) == list or type(lista) == ndarray) and len(lista) == 1:
        return flat(lista[0])
    return lista
                
    

def get_bloom_dataset(data_path):
    dataset = serialize.load_dataset(data_path)
    features = [el for el in dataset.columns if el!= 'url' and el != 'score']
    X = dataset[features].iloc[:,1:-1].to_numpy()
    y = dataset[features].iloc[:,-1].replace(-1, 0).to_numpy() # .replace(-1, 0) per binary loss
    url = dataset['url']
    X_train, _ ,y_train ,_ = train_test_split(X,y,train_size= 0.3, random_state=42)
    return (X_train, y_train, url, X,y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_list", action = "store", dest = "classifier_list", type = str, nargs = '+', required= True, help = "list of used classifier " )
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")    
    args = parser.parse_args()
    train_classifier(*get_bloom_dataset(args.data_path),args.classifier_list, args.data_path)

        # e train



    