import pandas as pd
import pickle 
import os


def load_dataset(path):
    return pd.read_csv(path)

def save_model(model, save_path):
    '''salva i modelli'''
    save_path += ".pk1"
    with open(save_path,'wb') as file:
            pickle.dump(model,file)
    size = os.path.getsize(save_path)
    return size

def save_score(model, X_test, url, save_path):
    '''salva i punteggi in un file csv, in modo da poterli poi utilizzare per la creazione di filtri'''         
    score = model.get_probs(X_test)
    save_score(save_path, score, url)
    d = {'url' : url, 'score' : score}
    save_object = pd.DataFrame(d)
    save_object.to_csv(save_path)
        


