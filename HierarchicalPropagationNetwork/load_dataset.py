import json

from networkx.readwrite import json_graph

from util import constants
from util.util import tweet_node

import os


'''
Questa funzione un grafo ad albero dal dato JSON utilizzando la funzione tree_graph dal modulo json_graph.
Identifica il nodo radice dell'albero cercando il nodo con un grado di ingresso pari a 0.
Costruisce il nodo di tweet utilizzando una funzione di aiuto di depth-first search (DFS).
MODIFICA: per la ricerca del nodo radice, invece di applicare la funzione "in_degree()" della classe nx.DiGraph() 
sul grafo "new_graph", si usa il metodo "in_degree()" sul grafo "new_graph" restituito dalla funzione "tree_graph"
'''
def construct_tweet_node_from_json(json_data):
    new_graph = json_graph.tree_graph(json_data)
    #root_node = [node for node, in_degree in nx.DiGraph.in_degree(new_graph).items() if in_degree == 0][0]
    root_node = [node for node, in_degree in new_graph.in_degree() if in_degree == 0][0]
    node_id_obj_dict = dict()
    dfs_node_construction_helper(root_node, new_graph, set(), node_id_obj_dict)
    return node_id_obj_dict[root_node]


'''
Questa è una funzione di supporto utilizzata per costruire ricorsivamente i nodi del grafo. Utilizza una visita in 
profondità per attraversare il grafo e costruire i nodi e gli archi associati.
MODIFICA: viene rimosso il type hinting (definizione esplicita del tipo degli argomenti e del valore di ritorno della
funzione)
'''
def dfs_node_construction_helper(node_id, graph, visited, node_id_obj_dict):
    if node_id in visited:
        return None

    visited.add(node_id)

    tweet_node_obj = construct_tweet_node_from_nx_node(node_id, graph)

    node_id_obj_dict[node_id] = tweet_node_obj

    for neighbor_node_id in graph.successors(node_id):
        if neighbor_node_id not in visited:
            dfs_node_construction_helper(neighbor_node_id, graph, visited, node_id_obj_dict)
            add_node_object_edge(node_id, neighbor_node_id, node_id_obj_dict)
'''
def dfs_node_construction_helper(node_id, graph: nx.DiGraph, visited: set, node_id_obj_dict: dict):
    if node_id in visited:
        return None

    visited.add(node_id)

    tweet_node_obj = construct_tweet_node_from_nx_node(node_id, graph)

    node_id_obj_dict[node_id] = tweet_node_obj

    for neighbor_node_id in graph.successors(node_id):
        if neighbor_node_id not in visited:
            dfs_node_construction_helper(neighbor_node_id, graph, visited, node_id_obj_dict)
            add_node_object_edge(node_id, neighbor_node_id, node_id_obj_dict)

'''


'''
Questa funzione aggiunge un arco tra due un nodo genitore e un nodo figlio, prendendo in considerazione il tipo di 
relazione tra i nodi nel dizionario 'node_id_obj_dict'.
'''
def add_node_object_edge(parent_node_id: int, child_node_id: int, node_id_obj_dict: dict):
    parent_node = node_id_obj_dict[parent_node_id]
    child_node = node_id_obj_dict[child_node_id]

    if child_node.node_type == constants.RETWEET_NODE:
        parent_node.add_retweet_child(child_node)
    elif child_node.node_type == constants.REPLY_NODE:
        parent_node.add_reply_child(child_node)
    else:
        # news node add both retweet and reply edge
        parent_node.add_retweet_child(child_node)
        parent_node.add_reply_child(child_node)

'''
Questa funzione prende un nodo di un grafo NetworkX e lo converte in un oggetto di tipo tweet_node.
MODIFICA: cambia il modo in cui si accede ai dati del nodo del grafo
'''
def construct_tweet_node_from_nx_node(node_id, graph):
    node_data = graph.nodes[node_id]
    return tweet_node(tweet_id=node_data['tweet_id'],
                      created_time=node_data['time'],
                      node_type=node_data['type'],
                      user_id=node_data['user'],
                      botometer_score=node_data.get('bot_score', None),
                      sentiment=node_data.get('sentiment', None))
'''
def construct_tweet_node_from_nx_node(node_id, graph: nx.DiGraph):
    return tweet_node(tweet_id=graph.node[node_id]['tweet_id'],
                      created_time=graph.node[node_id]['time'],
                      node_type=graph.node[node_id]['type'],
                      user_id=graph.node[node_id]['user'],
                      botometer_score=graph.node[node_id].get('bot_score', None),
                      sentiment=graph.node[node_id].get('sentiment', None))
'''

'''
Questa funzione restituisce una lista di ID di campioni dal file corrispondente nel dataset.
'''
def get_dataset_sample_ids(news_source, news_label, dataset_dir="data/sample_ids"):
    sample_list = []
    with open("{}/{}_{}_ids_list.txt".format(dataset_dir, news_source, news_label)) as file:
        for id in file:
            sample_list.append(id.strip())

    return sample_list

'''
Questa funzione carica i nodi del grafo NetworkX dai file JSON nel dataset, costruisce i nodi e gli archi e restituisce 
una lista di oggetti tweet_node.
'''
def load_from_nx_graphs(dataset_dir: str, news_source: str, news_label: str):
    tweet_node_objects = []

    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
        with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
            tweet_node_objects.append(construct_tweet_node_from_json(json.load(file)))

    return tweet_node_objects

'''
Questa funzione carica i grafi NetworkX dai file JSON nel dataset e restituisce una lista di grafi.
MODIFICA: load_networkx_graphs carica grafi completi in oggetti NetworkX (DiGraph), mentre il codice successivo si 
aspetta oggetti nodo con attributi specifici, come reply_children.
'''
def load_networkx_graphs(dataset_dir: str, news_source: str, news_label: str):
    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    news_samples = []

    for news_file in os.listdir(news_dataset_dir):
        #with open("{}/{}.json".format(news_dataset_dir, news_file)) as file:
        with open("{}/{}".format(news_dataset_dir, news_file)) as file:
            #news_samples.append(json_graph.tree_graph(json.load(file)))
            news_samples.append(construct_tweet_node_from_json(json.load(file)))

    return news_samples

'''
Questa funzione carica il dataset completo, compresi i campioni di notizie vere e false, utilizzando le funzioni sopra 
descritte.
'''
def load_dataset(dataset_dir: str, news_source: str):
    fake_news_samples = load_networkx_graphs(dataset_dir, news_source, "fake")
    real_news_samples = load_networkx_graphs(dataset_dir, news_source, "real")

    return fake_news_samples, real_news_samples


if __name__ == '__main__':
    pf_fake_samples, pf_real_samples = load_dataset("data/nx_network_data", "politifact")
    print("Dimensione fake news - Politifact:", len(pf_fake_samples))
    print("Dimensione real news - Politifact:", len(pf_real_samples))
    gc_fake_samples, gc_real_samples = load_dataset("data/nx_network_data", "gossipcop")
    print("Dimensione fake news - GossipCop:", len(gc_fake_samples))
    print("Dimensione real news - GossipCop:", len(gc_real_samples))
