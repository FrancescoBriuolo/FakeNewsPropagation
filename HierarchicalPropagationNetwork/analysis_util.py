import errno
import os
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.utils import resample

from stat_test import perform_t_test, get_box_plots_mod
from util.util import twitter_datetime_str_to_object

'''
Questa classe è progettata per aiutare nella gestione delle caratteristiche (features) utilizzate nell'analisi dei dati. 
Fornisce metodi per ottenere nomi delle caratteristiche, calcolare statistiche, e salvare e caricare dati delle 
caratteristiche.
'''
class BaseFeatureHelper(metaclass=ABCMeta):

    @abstractmethod
    def get_feature_group_name(self):
        pass

    @abstractmethod
    def get_micro_feature_method_references(self):
        pass

    @abstractmethod
    def get_micro_feature_method_names(self):
        pass

    @abstractmethod
    def get_micro_feature_short_names(self):
        pass

    @abstractmethod
    def get_macro_feature_method_references(self):
        pass

    @abstractmethod
    def get_macro_feature_method_names(self):
        pass

    @abstractmethod
    def get_macro_feature_short_names(self):
        pass

    def get_dump_file_name(self, news_source, micro_features, macro_features, label, file_dir):
        file_tags = [news_source, label, self.get_feature_group_name()]
        if micro_features:
            file_tags.append("micro")

        if macro_features:
            file_tags.append("macro")

        return "{}/{}.pkl".format(file_dir, "_".join(file_tags))

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None,
                           file_dir="data/features", use_cache=False):
        function_refs = []

        file_name = self.get_dump_file_name(news_source, micro_features, macro_features, label, file_dir)
        data_file = Path(file_name)

        if use_cache and data_file.is_file():
            return pickle.load(open(file_name, "rb"))

        if micro_features:
            function_refs.extend(self.get_micro_feature_method_references())

        if macro_features:
            function_refs.extend(self.get_macro_feature_method_references())

        if len(function_refs) == 0:
            return None

        all_features = []

        for function_reference in function_refs:
            features_set = get_sample_feature_value(prop_graphs, function_reference)
            all_features.append(features_set)

        feature_array = np.transpose(get_numpy_array(all_features))

        pickle.dump(feature_array, open(file_name, "wb"))

        return feature_array

    def get_feature_names(self, micro_features, macro_features):
        features_names = []
        short_feature_names = []

        if micro_features:
            features_names.extend(self.get_micro_feature_method_names())
            short_feature_names.extend(self.get_micro_feature_short_names())

        if macro_features:
            features_names.extend(self.get_macro_feature_method_names())
            short_feature_names.extend(self.get_macro_feature_short_names())

        return features_names, short_feature_names

    def print_statistics_for_all_features(self, feature_array=None, prop_graphs=None, micro_features=None,
                                          macro_features=None):

        if feature_array is None:
            feature_array = self.get_features_array(prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            feature_values = feature_array[:, idx]
            print_stat_values(feature_names[idx], feature_values, short_feature_names[idx])

    def save_box_plots_for_features(self, fake_feature_array=None, real_feature_array=None, fake_prop_graphs=None,
                                     real_prop_graphs=None, micro_features=None, macro_features=None, save_folder=None):

        if fake_feature_array is None:
            fake_feature_array = self.get_features_array(fake_prop_graphs, micro_features, macro_features)
            real_feature_array = self.get_features_array(real_prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]

            get_box_plots_mod(fake_feature_values, real_feature_values, save_folder, short_feature_names[idx])

    def get_feature_significance_t_tests(self, fake_feature_array, real_feature_array, micro_features=None,
                                         macro_features=None):
        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]
            print("Feature {} : {}".format(short_feature_names[idx], feature_names[idx]))
            perform_t_test(fake_feature_values, real_feature_values)

    def get_feature_significance_bootstrap_tests(self, fake_feature_array, real_feature_array, micro_features=None,
                                                 macro_features=None):

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]

            perms_fake = []
            perms_real = []

            combined = np.concatenate((fake_feature_values, real_feature_values), axis=0)

            print("combined shape : ", combined.shape)

            for i in range(10000):
                np.random.seed(i)
                perms_fake.append(resample(combined, n_samples=len(fake_feature_values)))
                perms_real.append(resample(combined, n_samples=len(real_feature_values)))

            dif_bootstrap_means = (np.mean(perms_fake, axis=1) - np.mean(perms_real, axis=1))
            print("diff bootstrap means : ", dif_bootstrap_means.shape)

            obs_difs = (np.mean(fake_feature_values) - np.mean(real_feature_values))

            p_value = dif_bootstrap_means[dif_bootstrap_means >= obs_difs].shape[0] / 10000

            print("Feature {} : {}".format(short_feature_names[idx], feature_names[idx]))
            print("t- value : {}   p-value : {}".format(obs_difs, p_value))

'''
Questa funzione prende una lista di grafi di notizie e una funzione di estrazione della caratteristica come input e 
restituisce una lista dei valori delle caratteristiche estratte da ciascun grafico.
'''
def get_sample_feature_value(news_graps: list, get_feature_fun_ref):
    result = []
    for graph in news_graps:
        result.append(get_feature_fun_ref(graph))

    return result

'''
Questa funzione controlla se una directory esiste e, se non esiste, la crea.
'''
def  create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Cartella "+dir_name+" creata con successo")
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        print("La cartella già esiste")


'''
Queste funzioni (get_epoch_timestamp_from_retweet e sort_retweet_object_by_time) sono utilizzate per ordinare gli 
oggetti di ritweet in base al tempo di creazione.
'''
def get_epoch_timestamp_from_retweet(retweet):
    return twitter_datetime_str_to_object(retweet["created_at"])


def sort_retweet_object_by_time(retweets: list):
    retweets.sort(key=get_epoch_timestamp_from_retweet)

    return retweets

'''
Questa funzione carica gli ID delle notizie da ignorare dal file data/news_id_ignore_list.
'''
def get_noise_news_ids():
    with open("data/news_id_ignore_list") as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

'''
Queste funzioni (load_prop_graph e remove_prop_graph_noise) sono utilizzate per caricare i grafi di propagazione delle 
notizie e rimuovere i grafi rumorosi.
'''
def load_prop_graph(data_folder, news_source, news_label):
    news_graphs = pickle.load(open("{}/{}_{}_news_prop_graphs.pkl".format(data_folder, news_source, news_label), "rb"))
    return news_graphs


def remove_prop_graph_noise(news_graphs, noise_ids):
    noise_ids = set(noise_ids)
    return [graph for graph in news_graphs if graph.tweet_id not in noise_ids]

'''
Questa funzione ordina gli oggetti di nodo tweet in base al tempo di creazione.
'''
def sort_tweet_node_object_by_created_time(tweet_nodes: list):
    tweet_nodes.sort(key=lambda x: x.created_time)

    return tweet_nodes

'''
Questa funzione prende due campioni e li riduce entrambi alla stessa lunghezza, mescolandoli casualmente.
'''
def equal_samples(sample1, sample2):
    target_len = min(len(sample1), len(sample2))

    np.random.seed(0)

    np.random.shuffle(sample1)
    np.random.shuffle(sample2)

    return sample1[:target_len], sample2[:target_len]


# def get_propagation_graphs(data_folder, news_source):
#     fake_propagation_graphs = load_prop_graph(data_folder, news_source, "fake")
#     real_propagation_graphs = load_prop_graph(data_folder, news_source, "real")
#
#     print("Before filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
#     print("Before filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
#
#     fake_propagation_graphs = remove_prop_graph_noise(fake_propagation_graphs, get_noise_news_ids())
#     real_propagation_graphs = remove_prop_graph_noise(real_propagation_graphs, get_noise_news_ids())
#
#     print("After filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
#     print("After filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
#     print(flush=True)
#
#     return fake_propagation_graphs, real_propagation_graphs

'''
Questa funzione converte una lista di liste in un array NumPy.
'''
def get_numpy_array(list_of_list):
    np_array_lists = []
    for list_obj in list_of_list:
        np_array_lists.append(np.array(list_obj))

    return np.array(np_array_lists)

'''
Questa funzione stampa statistiche di base (minimo, massimo, media) per un insieme di valori.
'''
def print_stat_values(feature_name, values, short_feature_name=""):
    print("=========================================")
    print("Feature {} : {}".format(short_feature_name, feature_name))
    print("Min value : {}".format(min(values)))
    print("Max value : {}".format(max(values)))
    print("Mean value : {}".format(np.mean(np.array(values))))
    print("=========================================")
