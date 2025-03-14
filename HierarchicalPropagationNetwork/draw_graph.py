import json

import networkx as nx
from matplotlib import pyplot as plt
from networkx.readwrite import json_graph

from util import constants
from util.util import tweet_node

import os


'''
Questa funzione crea un grafo ad albero dal dato JSON utilizzando la funzione tree_graph dal modulo json_graph.
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
    #return node_id_obj_dict[root_node], node_id_obj_dict
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
Questa funzione carica i nodi del grafo NetworkX dai file JSON presenti in una specifica directory del dataset. 
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
'''
def load_networkx_graphs(dataset_dir: str, news_source: str, news_label: str):
    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    news_samples = []

    for news_file in os.listdir(news_dataset_dir):
        #with open("{}/{}.json".format(news_dataset_dir, news_file)) as file:
        with open("{}/{}".format(news_dataset_dir, news_file)) as file:
            news_samples.append(json_graph.tree_graph(json.load(file)))

    return news_samples

'''
Questa funzione carica il dataset completo, compresi i campioni di notizie vere e false, utilizzando le funzioni sopra 
descritte.
'''
def load_dataset(dataset_dir: str, news_source: str):
    fake_news_samples = load_networkx_graphs(dataset_dir, news_source, "fake")
    real_news_samples = load_networkx_graphs(dataset_dir, news_source, "real")

    return fake_news_samples, real_news_samples

def draw_graph(graph):
    node_colors = []
    node_labels = {}
    edge_colors = []

    for node_id, data in graph.nodes(data=True):
        if data['type'] == constants.NEWS_ROOT_NODE:
            node_colors.append('green')  # Nodi news_node
        elif data['type'] == constants.POST_NODE:
            node_colors.append('pink')  # Nodi tweet_node
        elif data['type'] == constants.RETWEET_NODE:
            node_colors.append('blue')  # Nodi retweet_node
        elif data['type'] == constants.REPLY_NODE:
            node_colors.append('magenta')  # Nodi reply_node
        node_labels[node_id] = str(node_id)

    macro_level_edges = []  # Definisci la rete macro-livello

    for u, v, data in graph.edges(data=True):
        if data['macro_level']:
            edge_colors.append('gray')  # Archi di rete macro-livello
        else:
            if graph.nodes[u]['type'] == constants.RETWEET_NODE:
                edge_colors.append('gray')  # Archi relativi a retweet
            else:
                edge_colors.append('black')  # Altrimenti archi neri

    pos = nx.spring_layout(graph)  # Posizionamento del grafo

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)

    # Aggiungi la legenda
    legend_labels = {
        'News Node': 'green',
        'Tweet Node': 'pink',
        'Retweet Node': 'blue',
        'Reply Node': 'magenta',
        'Macro-level Edge': 'gray',
        'Other Edge': 'black'
    }
    plt.legend(handles=[
        plt.Line2D([], [], color=color, label=label)
        for label, color in legend_labels.items()
    ])

    plt.show()


def print_node_info(node):
    # Stampa le informazioni del nodo corrente
    contents = node.get_contents()
    for key, value in contents.items():
        print(f"{key}: {value}")

    # Se il nodo ha figli retweettati, stampa le informazioni su di essi
    if node.retweet_children:
        print("\nRetweet Children:")
        for child in node.retweet_children:
            print_node_info(child)
            print()

    # Se il nodo ha figli che hanno risposto, stampa le informazioni su di essi
    if node.reply_children:
        print("\nReply Children:")
        for child in node.reply_children:
            print_node_info(child)
            print()


# Funzione per costruire il grafo a partire dal nodo radice e etichettare i nodi con tweet_id
def build_graph_from_json(json_data):
    # Costruisci il grafo
    G = nx.DiGraph()

    # Funzione ricorsiva per aggiungere nodi e archi al grafo
    def add_node_to_graph(node):
        # Estrai il campo node_type dal nodo
        node_type = node.node_type

        # Aggiungi il nodo corrente al grafo e assegna il colore in base al tipo del nodo
        if node_type == 1:
            G.add_node(node.tweet_id, color='green')
        elif node_type == 2:
            G.add_node(node.tweet_id, color='pink')
        elif node_type == 3:
            G.add_node(node.tweet_id, color='blue')
        elif node_type == 4:
            G.add_node(node.tweet_id, color='magenta')

        # Aggiungi gli archi ai nodi figli
        for child in node.retweet_children + node.reply_children:
            G.add_edge(node.tweet_id, child.tweet_id)
            add_node_to_graph(child)


    # Aggiungi il nodo radice e costruisci il resto del grafo
    add_node_to_graph(json_data)

    return G

def draw_colored_graph(graph):
    # Estrai i colori dei nodi dal grafo
    node_colors = [data['color'] for _, data in graph.nodes(data=True)]

    # Posiziona i nodi utilizzando l'algoritmo spring_layout
    pos = nx.spring_layout(graph)

    # Disegna il grafo
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='black')

    # Aggiungi una legenda per i colori dei nodi
    legend_labels = {
        'News Node': 'green',
        'Tweet Node': 'pink',
        'Retweet Node': 'blue',
        'Reply Node': 'magenta'
    }
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', linestyle='None', color=color, label=label)
        for label, color in legend_labels.items()
    ])

    # Mostra il grafo
    plt.show()

if __name__ == '__main__':

    # Lettura del file JSON
    with open(
            "C:/Users/franc/PycharmProjects/Statistiche/fake-news-propagation/data/nx_network_data/politifact_fake/politifact13038.json",
            "r") as json_file:
        json_data = json.load(json_file)

        # Costruzione del nodo radice tweet_node
        root_node = construct_tweet_node_from_json(json_data)

        # Costruzione del grafo
        graph = build_graph_from_json(root_node)

        # Stampa delle informazioni del grafo
        print("Nodi del grafo:")
        print(list(graph.nodes()))

        # Disegno del grafo colorato
        draw_colored_graph(graph)