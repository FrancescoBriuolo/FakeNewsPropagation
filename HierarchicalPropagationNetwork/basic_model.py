import time
from sklearn.inspection import permutation_importance

import matplotlib
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from construct_sample_features import get_TPNF_dataset, get_train_test_split, get_dataset_feature_names
from analysis_util import create_dir

matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys

import os


'''
Questa funzione restituisce un classificatore basato sul nome specificato come parametro.
'''
def get_classifier_by_name(classifier_name):
    if classifier_name == "GaussianNB":
        return GaussianNB()
    elif classifier_name == "LogisticRegression":
        #return LogisticRegression(solver='lbfgs', random_state=0)
        return LogisticRegression(solver='lbfgs')
    elif classifier_name == "DecisionTreeClassifier": #criterion, max_depth, max_features
        #return DecisionTreeClassifier(random_state=0)
        return DecisionTreeClassifier()
    elif classifier_name == "RandomForestClassifier": #n_estimators, criterion, max_depth, max_features
        #return RandomForestClassifier(n_estimators=50, random_state=0)
        #return RandomForestClassifier(n_estimators=50)
        return RandomForestClassifier()
    elif classifier_name == "SVM -linear kernel":
        #return svm.SVC(kernel='linear', random_state=0)
        #return svm.SVC(kernel='linear') #C, kernel
        return svm.SVC()
    elif classifier_name == "FNN":  #hidden_layer_sizes. activation. batch_size. learning_rate
        return MLPClassifier()
        #hidden_layer_sizes=[10], activation='relu', max_iter=1000, batch_size=8,
                             #early_stopping=True, n_iter_no_change=15, verbose=True, learning_rate='adaptive')
    #NN2 era il migliore
    elif classifier_name == "NN2":
        return MLPClassifier(hidden_layer_sizes=[20], activation='relu', max_iter=1000, batch_size=8,
                             early_stopping=True, n_iter_no_change=15, verbose=True, learning_rate='adaptive')
    elif classifier_name == "NN":  #hidden_layer_sizes. activation. batch_size. learning_rate
        return MLPClassifier()

'''
Restituisce il dizionario di parametri con cui effettuare il tuning.
'''
def get_params_by_name(classifier_name):
    if classifier_name == "GaussianNB":
        return {
            # Controlla la quantità di smoothing applicata alla varianza
            'var_smoothing': [1e-9]
        }
    elif classifier_name == "LogisticRegression":
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'max_iter': [500]
        }
    elif classifier_name == "DecisionTreeClassifier":
        #return DecisionTreeClassifier(random_state=0)
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'max_features': [None, 'sqrt', 'log2']
        }
    elif classifier_name == "RandomForestClassifier":
        #return RandomForestClassifier(n_estimators=50, random_state=0)
        #return RandomForestClassifier(n_estimators=50)
        return {
            'criterion': ['gini', 'entropy'],       # Funzione di misura dell'impurità
            'max_depth': [None, 10, 20, 30, 40, 50],# Profondità massima degli alberi
            'max_features': [None, 'sqrt', 'log2'],# Numero di caratteristiche da considerare quando si cerca la divisione migliore
            'n_estimators': [10, 20, 50, 100],  # Numero di alberi nella foresta
        }
    elif classifier_name == "SVM -linear kernel":
        #return svm.SVC(kernel='linear', random_state=0)
        #return svm.SVC(kernel='linear')
        return {
            'C': [0.1, 1, 10, 100, 1000],                   # Parametro di regolarizzazione
            'kernel': ['linear'],                           # Tipo di kernel da utilizzare nel calcolo + , 'poly', 'rbf', 'sigmoid'
            'max_iter': [2000]
        }
    elif classifier_name == "FNN":
        return {
            'hidden_layer_sizes': [[10], [15], [20], [40], [10,10], [20,10], [50], [60], [70],
                                [10, 10, 10], [20, 20, 20], [50, 50, 50], [70, 70, 70], [80],
                                [10, 20], [20, 10, 20], [50, 20, 10], [80, 80, 80],
                                [100], [100, 50], [100, 50, 25], [110],
                                [10, 50, 10], [50, 10, 50], [100, 50, 10]],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [2000],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [8, 16, 32, 64],
            'early_stopping': [True],
            'n_iter_no_change': [15]
        }
    elif classifier_name == "NN2":
        return {
            'hidden_layer_sizes': [20],
            'activation':['relu'],
            'max_iter':[1000],
            'batch_size':[8],
            'early_stopping':[True],
            'n_iter_no_change':[15],
            'verbose':[False],
            'learning_rate':['adaptive']
        }
    elif classifier_name == "NN":
        return {
            'hidden_layer_sizes': [50],
            'activation':['tanh'],
            'max_iter':[2000],
            'solver': ['adam'],
            'batch_size':[8],
            'early_stopping':[True],
            'n_iter_no_change':[15],
            'verbose':[False],
            'learning_rate':['constant']
        }

'''
Questa funzione addestra il classificatore specificato sui dati di addestramento e valuta le prestazioni utilizzando 
i dati di test. Le prestazioni del modello sono valutate utilizzando diverse metriche come accuracy, precision, recall 
e score F1.
'''
def train_model(news_source, features, classifier_name, feature_names, X_train, X_test, y_train, y_test):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    classifier = get_classifier_by_name(classifier_name)
    params = get_params_by_name(classifier_name)

    print("Classificatore:", classifier)  # Questo dovrebbe stampare un oggetto classificatore, non None
    print("Parametri:", params)  # Questo dovrebbe stampare un dizionario, non None

    grid_search = GridSearchCV(get_classifier_by_name(classifier_name), get_params_by_name(classifier_name), cv=5,
                               verbose=2)

    pca = None


    # decommenta le prossime 4 righe per attivare la pca
    # from sklearn.decomposition import PCA
    # pca = PCA(0.95)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print('Best params for', classifier_name, best_params)

    create_dir(f'data/metrics/{news_source}_metrics/{features}')

    output_file = f'data/metrics/{news_source}_metrics/{features}/{classifier_name}_best_params.txt'
    with open(output_file, 'a') as file:
        file.write(f'Best params for {classifier_name}: {best_params}\n')



    importances = None
    importances_std = None
    for i in range(5):
        classifier_clone = get_classifier_by_name(classifier_name)
        classifier_clone.set_params(**grid_search.best_params_)
        classifier_clone.fit(X_train, y_train)

        predicted_output = classifier_clone.predict(X_test)
        accuracy, precision, recall, f1_score_val = get_metrics(y_test, predicted_output, one_hot_rep=False)

        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1_score_val)

        result = permutation_importance(classifier_clone, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        if importances is None:
            importances = result.importances_mean/5
            importances_std = result.importances_std/5
        else:
            importances += result.importances_mean/5
            importances_std += result.importances_std/5
    #create_dir(f'data/metrics/{news_source}_metrics/{features}')

    with open(f'data/metrics/{news_source}_metrics/{features}/{classifier_name}_metrics.txt', 'w') as f:
        print_metrics(
            round(np.mean(accuracy_values), 3),
            round(np.mean(precision_values), 3),
            round(np.mean(recall_values), 3),
            round(np.mean(f1_score_values), 3),
            file=f
        )

    with open(f'data/metrics/{news_source}_metrics/{features}/{classifier_name}_std_metrics.txt', 'w') as f:
        print_metrics(
            round(np.std(accuracy_values), 3),
            round(np.std(precision_values), 3),
            round(np.std(recall_values), 3),
            round(np.std(f1_score_values), 3),
            file=f
        )

    if pca is None:
        # Plot the feature importances
        sorted_idx = np.abs(importances).argsort()
        print('sorted_idx:', sorted_idx)

        # Sort the features by importance
        sorted_features = [feature_names[idx] for idx in sorted_idx]
        print('sorted_features:', sorted_features)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_idx)), importances[sorted_idx], yerr=importances_std[sorted_idx], align='center')
        plt.xticks(range(len(sorted_idx)), [f'{feature_names[i]}' for i in sorted_idx]) #, rotation=90
        plt.ylabel("Permutation Importance")
        plt.title(f'Permutation Importances for {classifier_name}')
        plt.tight_layout()
        plt.savefig(f'data/metrics/{news_source}_metrics/{features}/{classifier_name}_importances.png')
        plt.close()
'''
Questa funzione stampa le metriche di valutazione delle prestazioni del modello.
'''
def print_metrics(accuracy, precision, recall, f1_score_val, file=sys.stdout):
    print("Accuracy : {}".format(accuracy), file=file)
    print("Precision : {}".format(precision), file=file)
    print("Recall : {}".format(recall), file=file)
    print("F1 : {}".format(f1_score_val), file=file)

'''
Questa funzione calcola diverse metriche di valutazione delle prestazioni del modello, come accuracy, precision, 
recall e score F1.
RIVEDERE
'''
def get_metrics(target, logits, one_hot_rep=True):
    """
    Two numpy one hot arrays
    :param target:
    :param logits:
    :return:
    """

    if one_hot_rep:
        label = np.argmax(target, axis=1)
        predict = np.argmax(logits, axis=1)
    else:
        label = target
        predict = logits

    accuracy = accuracy_score(label, predict)

    precision = precision_score(label, predict)
    recall = recall_score(label, predict)
    f1_score_val = f1_score(label, predict)

    return accuracy, precision, recall, f1_score_val

'''
Questa funzione addestra e valuta modelli di classificazione di base come Gaussian Naive Bayes, Logistic Regression, 
Decision Tree, Random Forest e SVM utilizzando dati di addestramento e test.
RIVEDERE
'''
def get_basic_model_results(news_source, features, feature_names, X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)


    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier_names = ["GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
                        "SVM -linear kernel"]
    #classifier_names = ["LogisticRegression"]
    classifier_names = ["FNN"]


    for idx in range(len(classifier_names)):
        #print("======={}=======".format(classifier_names[idx]))
        train_model(news_source, features, classifier_names[idx], feature_names, X_train, X_test, y_train, y_test)

def save_in_CSV(data_list, output_dir, file_name):
    # Crea il percorso completo del file CSV_complete_dataset
    csv_file_path = os.path.join(output_dir, file_name + ".csv")

    # Crea un DataFrame pandas dalla lista di dati
    df = pd.DataFrame(data_list)

    # Salva il DataFrame nel file CSV_complete_dataset
    df.to_csv(csv_file_path, index=False)

    print("Dati salvati con successo in:", csv_file_path)

'''
Questa funzione carica il dataset di caratteristiche, addestra e valuta i modelli di classificazione utilizzando le 
funzioni sopra descritte. I dati di classificazione sono specifici per le notizie vere o false (fake news) e vengono 
generati per un intervallo di tempo specificato.
RIVEDERE
'''
def get_classificaton_results_tpnf(data_dir, news_source, time_interval, use_cache=False):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = True

    sample_feature_array, file_name = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic, time_interval, use_cache=use_cache)

    feature_names, short_feature_names = get_dataset_feature_names(include_micro, include_macro, include_structural,
                                                                   include_temporal, include_linguistic)
    print("Feature names:", feature_names)
    print("Short feature names:", short_feature_names)

    print("FILE NAME", file_name)
    output_dir = os.path.join("data/CSV", news_source)
    save_in_CSV(sample_feature_array, output_dir, file_name)

    features = "_".join(file_name.split("_")[1:])
    print("FEATURES", features)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    print("Lunghezza sample_feature_array:", len(sample_feature_array))
    print("Lunghezza target_labels:", len(target_labels))

    # Verifica che le lunghezze corrispondano
    if len(sample_feature_array) != len(target_labels):
        raise ValueError("Inconsistent number of samples between features and target labels")

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(news_source, features, short_feature_names, X_train, X_test, y_train, y_test)


'''
Questa funzione traccia l'importanza delle caratteristiche ottenute da un modello di classificazione.
RIVEDERE OUTPUT DIR
'''
def plot_feature_importances(coef, names, output_dir):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)

    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()
    #plt.show()

'''
Questa funzione calcola e salva l'importanza delle caratteristiche ottenute da un modello Random Forest.
'''
def dump_random_forest_feature_importance(data_dir, news_source):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = True

    sample_feature_array = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic, use_cache=True)

    sample_feature_array = sample_feature_array[:, :-1]
    feature_names, short_feature_names = get_dataset_feature_names(include_micro, include_macro, include_structural,
                                                                   include_temporal, include_linguistic)

    feature_names = feature_names[:-1]
    short_feature_names = short_feature_names[:-1]
    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=100, random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    matplotlib.rcParams['figure.figsize'] = 5, 2

    # Plot the feature importances of the forest
    plt.figure()

    plt.bar(range(X_train.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), np.array(short_feature_names)[indices], rotation=75, fontsize=9.5)
    plt.xlim([-1, X_train.shape[1]])
    # Imposta manualmente i valori degli intervalli sull'asse y
    plt.yticks(np.arange(0, 0.200, 0.025))
    plt.savefig('{}_feature_importance.png'.format(news_source), bbox_inches='tight')
    plt.close()

    print("AAAAAAAAAAAAAAAAAaaaaaa")
    output_dir = os.path.join(f'data/feature_importance/{news_source}')
    plot_feature_importances(importances, short_feature_names, output_dir)
    print("BBBBBBBBBBBBBBBBBbbbbbb")


    #plt.show()

'''
Questa funzione esegue l'analisi dei dati di classificazione delle notizie vere e false in base agli intervalli di 
tempo specificati.
'''
def get_classificaton_results_tpnf_by_time(news_source: str):
    # Time Interval in hours for early-fake news detection
    time_intervals = [3, 6, 12, 24, 36, 48, 60, 72, 84, 96]

    for time_interval in time_intervals:
        print("=============Time Interval : {}  ==========".format(time_interval))
        start_time = time.time()
        get_classificaton_results_tpnf("data/features", news_source, time_interval)

        print("\n\n================Exectuion time - {} ==================================\n".format(
            time.time() - start_time))


if __name__ == "__main__":
    # Check sulla directory di lavoro
    #check_directory_existence("data/features")
    create_dir("data/features")

    #get_classificaton_results_tpnf("data/features", "politifact", time_interval=None, use_cache=False)
    get_classificaton_results_tpnf("data/features", "gossipcop", time_interval=None, use_cache=False)

    #dump_random_forest_feature_importance("data/features_RF", "politifact")

    # Filter the graphs by time interval (for early fake news detection) and get the classification results
    #get_classificaton_results_tpnf_by_time("politifact")
    #get_classificaton_results_tpnf_by_time("gossipcop")
