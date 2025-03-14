import pickle
import queue
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from analysis_util import equal_samples
from linguistic_analysis import get_all_linguistic_features, LinguisticFeatureHelper
from load_dataset import load_from_nx_graphs, load_networkx_graphs

from structure_temp_analysis import get_all_structural_features, StructureFeatureHelper, get_first_post_time
from temporal_analysis import get_all_temporal_features, TemporalFeatureHelper
from util.util import tweet_node

import os


'''
Funzione che verifica l'esistenza della directory di cui viene specificato il percorso. Se non esiste, la crea.
'''
'''
def check_directory_existence(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory creata")

        # Carica la lista dei file JSON da sample_ids/politifact_fake_ids_list.txt
        with open("data/sample_ids/politifact_fake_ids_list.txt", "r") as file:
            file_ids = file.read().splitlines()

        # Copia i file JSON dalla directory originale a quella specificata
        source_dir = "data/nx_network_data/politifact_fake"
        for file_id in file_ids:
            source_file = os.path.join(source_dir, f"{file_id}.json")
            destination_file = os.path.join(directory_path, f"{file_id}.json")
            shutil.copyfile(source_file, destination_file)

    else:
        print("Directory esistente")
'''


def check_directory_existence(base_path, news_source, news_label):
    directory_path = "{}/{}_{}".format(base_path, news_source, news_label)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory creata")

        # Carica la lista dei file JSON da sample_ids/<news_source>_<label>_ids_list.txt
        with open("data/sample_ids/{}_{}_ids_list.txt".format(news_source, news_label), "r") as file:
            file_ids = file.read().splitlines()

        # Copia i file JSON dalla directory originale a quella specificata
        source_dir = "data/nx_network_data/{}_{}".format(news_source, news_label)
        for file_id in file_ids:
            source_file = os.path.join(source_dir, f"{file_id}.json")
            destination_file = os.path.join(directory_path, f"{file_id}.json")
            shutil.copyfile(source_file, destination_file)
    else:
        print("Directory esistente")

'''
Questa funzione combina le caratteristiche temporali, strutturali e linguistiche di un insieme di grafi di 
propagazione di notizie per creare un array di feature per l'addestramento di modelli di machine learning.
'''
def get_features(news_graphs, micro_features, macro_features):
    temporal_features = get_all_temporal_features(news_graphs, micro_features, macro_features)
    structural_features = get_all_structural_features(news_graphs, micro_features, macro_features)
    linguistic_features = get_all_linguistic_features(news_graphs, micro_features, macro_features)

    sample_features = np.concatenate([temporal_features, structural_features, linguistic_features], axis=1)
    return sample_features

'''
Questa funzione carica o genera un dataset di propagazione di notizie da fonti come "Politifact" e "Gossipcop" con le 
relative etichette di target (fake o real) per l'addestramento di modelli.
'''
def get_dataset(news_source, load_dataset=False, micro_features=True, macro_features=True):
    if load_dataset:
        sample_features = pickle.load(open("{}_samples_features.pkl".format(news_source), "rb"))
        target_labels = pickle.load(open("{}_target_labels.pkl".format(news_source), "rb"))

    else:
        fake_prop_graph, real_prop_graph = get_nx_propagation_graphs(news_source)
        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

        print("fake samples len : {} real samples len : {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_news_samples = get_features(fake_prop_graph, micro_features, macro_features)
        real_news_samples = get_features(real_prop_graph, micro_features, macro_features)

        print("Fake feature array ")
        print(fake_news_samples.shape)

        print("real feature array")
        print(real_news_samples.shape)

        sample_features = np.concatenate([fake_news_samples, real_news_samples], axis=0)
        target_labels = np.concatenate([np.ones(len(fake_news_samples)), np.zeros(len(real_news_samples))], axis=0)

        pickle.dump(sample_features, (open("{}_samples_features.pkl".format(news_source), "wb")))
        pickle.dump(target_labels, (open("{}_target_labels.pkl".format(news_source), "wb")))

    return sample_features, target_labels

'''
Questa funzione suddivide il dataset in set di addestramento e test per l'addestramento e la valutazione dei modelli di 
machine learning.
'''
def get_train_test_split(samples_features, target_labels):
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels, stratify=target_labels,
                                                        test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

'''
Questa funzione esegue l'analisi delle componenti principali (PCA) sui dati di addestramento per ridurre la 
dimensionalità delle feature.
'''
def perform_pca(train_data, target_labels):
    pca = PCA(n_components=min(20, len(train_data[0])))
    pca.fit(train_data, target_labels)
    return pca

'''
Questa funzione restituisce il nome del file per il dataset specificato, in base alle feature selezionate.
'''
def get_dataset_file_name(file_dir, news_source, include_micro=True, include_macro=True, include_structural=True,
                          include_temporal=True,
                          include_linguistic=True):
    file_names = [news_source]
    if include_micro:
        file_names.append("micro")

    if include_macro:
        file_names.append("macro")

    if include_structural:
        file_names.append("struct")

    if include_temporal:
        file_names.append("temp")

    if include_linguistic:
        file_names.append("linguistic")

    return "{}/{}.pkl".format(file_dir, "_".join(file_names))

'''
Questa funzione genera un dataset con feature specifiche selezionate dall'utente, come micro, macro, strutturali, 
temporali e linguistiche.
'''
def get_TPNF_dataset(out_dir, news_source, include_micro=True, include_macro=True, include_structural=None,
                     include_temporal=None,
                     include_linguistic=None, time_interval=None, use_cache=False):
    file_name = get_dataset_file_name(out_dir, news_source, include_micro, include_macro, include_structural,
                                      include_temporal, include_linguistic)

    data_file = Path(file_name)

    # Controlla se è specificato di utilizzare la cache dei dati. Se sì e i dati sono già stati salvati in precedenza in
    # un file, carica semplicemente il file di dati e lo restituisce.
    if use_cache and data_file.is_file():
        return pickle.load(open(file_name, "rb"))

    # Se non è specificato di utilizzare la cache o se i dati non sono stati salvati in precedenza, chiama la funzione
    # get_dataset_feature_array per ottenere le caratteristiche del campione, sia per le notizie false che per quelle
    # vere, inclusi i micro e macro livelli di propagazione, e le caratteristiche strutturali, temporali e linguistiche.
    else:
        fake_sample_features, real_sample_features = get_dataset_feature_array(news_source, include_micro,
                                                                               include_macro, include_structural,
                                                                               include_temporal, include_linguistic,
                                                                               time_interval)
        print("Lunghezza fake_sample_features", len(fake_sample_features))
        print("Lunghezza real_sample_features", len(real_sample_features))
        print(fake_sample_features)

        # Concatena le caratteristiche dei campioni delle notizie false e vere in un unico array.
        sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        # Salva l'array dei campioni nel file specificato nella directory di output utilizzando la libreria pickle.
        pickle.dump(sample_features, open(file_name, "wb"))

        # Stampare il contenuto del file appena creato
        print("Contenuto del file {}: {}".format(file_name, sample_features))

        # Restituisce l'array dei campioni contenente le caratteristiche della propagazione delle notizie.
        return sample_features, os.path.splitext(os.path.basename(file_name))[0]


'''
Questa funzione restituisce i nomi delle feature utilizzate nel dataset, in base alle feature selezionate.
'''
def get_dataset_feature_names(include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None):
    feature_helpers = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())

    feature_names_all = []
    short_feature_names_all = []

    for idx, feature_helper in enumerate(feature_helpers):
        features_names, short_feature_names = feature_helper.get_feature_names(include_micro, include_macro)

        feature_names_all.extend(features_names)
        short_feature_names_all.extend(short_feature_names)

    return feature_names_all, short_feature_names_all

'''
Questa funzione controlla se un grafo di propagazione di notizie ha almeno un retweet o una risposta.
'''
def is_valid_graph(prop_graph: tweet_node, retweet=True, reply=True):
    """ Check if the prop graph has alteast one retweet or reply"""

    for post_node in prop_graph.children:
        if (retweet and len(post_node.reply_children) > 0) or (reply and len(post_node.retweet_children) > 0):
            return True

    return False

'''
Questa funzione rimuove i nodi di un grafo di propagazione di notizie che superano un limite di tempo specificato.
'''
def remove_node_by_time(graph: tweet_node, limit_time):
    start_time = get_first_post_time(graph)
    end_time = start_time + limit_time

    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.children

        retweet_children = set(node.retweet_children)
        reply_children = set(node.reply_children)

        for child in children.copy():

            if child.created_time <= end_time:
                q.put(child)
            else:
                node.children.remove(child)
                try:
                    retweet_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass
                try:
                    reply_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass

        node.retweet_children = list(retweet_children)
        node.reply_children = list(reply_children)

    return graph

'''
Questa funzione filtra i grafici di propagazione di notizie in base a un limite di tempo specificato e verifica se il 
grafo risultante è valido.
'''
def filter_propagation_graphs(graphs, limit_time):
    result_graphs = []

    for prop_graph in graphs:
        filtered_prop_graph = remove_node_by_time(prop_graph, limit_time)
        if is_valid_graph(filtered_prop_graph):
            result_graphs.append(filtered_prop_graph)

    return result_graphs

'''
Questa funzione carica i grafi di propagazione di notizie da file di dati salvati.
'''
def get_nx_propagation_graphs(data_folder, news_source, full=False):

    if full:
        fake_propagation_graphs = load_networkx_graphs(data_folder, news_source, "fake")
        real_propagation_graphs = load_networkx_graphs(data_folder, news_source, "real")
    else:
        fake_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "fake")
        real_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "real")

    return fake_propagation_graphs, real_propagation_graphs

'''
Questa funzione genera un array di feature per il dataset, utilizzando feature selezionate come micro, macro, 
strutturali, temporali e linguistiche.
'''
def get_dataset_feature_array(news_source, include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None, time_interval=None):

    print('get_dataset_feature_array')
    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source, full=True)


    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    if time_interval is not None:
        time_limit = time_interval * 60 * 60

        print("Time limit in seconds : {}".format(time_limit))

        fake_prop_graph = filter_propagation_graphs(fake_prop_graph, time_limit)
        real_prop_graph = filter_propagation_graphs(real_prop_graph, time_limit)

        print("After time based filtering ")
        print("No. of fake samples : {}  No. of real samples: {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = []
    feature_group_names = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())
        feature_group_names.append("Structural")

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())
        feature_group_names.append("Temporal")

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())
        feature_group_names.append("Linguistic")

    fake_feature_all = []
    real_feature_all = []
    for idx, feature_helper in enumerate(feature_helpers):
        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="real")

        feature_names = feature_helper.get_feature_names(micro_features=include_micro, macro_features=include_macro)
        print(feature_names)
        if fake_features is not None and real_features is not None:
            fake_feature_all.append(fake_features)
            real_feature_all.append(real_features)

            print("Feature group : {}".format(feature_group_names[idx]))
            print(len(fake_features))
            print(len(real_features), flush=True)

    return np.concatenate(fake_feature_all, axis=1), np.concatenate(real_feature_all, axis=1)

'''
Questa funzione calcola e stampa le statistiche del dataset, inclusi test t, boxplot e altre statistiche descrittive, 
per valutare le feature e la loro rilevanza per l'addestramento del modello.
'''
def get_dataset_statistics(news_source):
    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/saved_new_no_filter", news_source)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = [StructureFeatureHelper(), TemporalFeatureHelper(), LinguisticFeatureHelper()]
    feature_group_names = ["StructureFeatureHelper", "TemporalFeatureHelper", "LinguisticFeatureHelper"]

    for idx, feature_helper in enumerate(feature_helpers):
        print("Feature group : {}".format(feature_group_names[idx]))

        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="real")

        feature_helper.save_box_plots_for_features(fake_feature_array=fake_features, real_feature_array=real_features,
                                                    save_folder="data/feature_images/{}".format(news_source),
                                                    micro_features=True, macro_features=True)

        feature_helper.get_feature_significance_t_tests(fake_features, real_features, micro_features=True,
                                                        macro_features=True)

        # Print the statistics of the dataset
        print("------------Fake------------")
        feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

        print("------------Real------------")
        feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)


if __name__ == "__main__":
    # Check sulla directory di lavoro
    check_directory_existence("data/saved_new_no_filter", "politifact", "fake")
    check_directory_existence("data/saved_new_no_filter", "politifact", "real")
    check_directory_existence("data/saved_new_no_filter", "gossipcop", "fake")
    check_directory_existence("data/saved_new_no_filter", "gossipcop", "real")
    get_dataset_statistics("politifact")
    get_dataset_statistics("gossipcop")
