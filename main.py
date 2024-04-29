import networkx as nx
import matplotlib.pyplot as plt

import csv
from datetime import datetime


def categorize_label(label):
    if label == 'true':
        return 0
    elif label == 'mostly-true':
        return 0.2
    elif label == 'half-true':
        return 0.4
    elif label == 'barely-true':
        return 0.6
    elif label == 'false':
        return 0.8
    elif label == 'pants-fire':
        return 1
    else:
        return None

'''
def extract_fields_from_tsv(file_path):
    speakers = set()
    topics = set()
    speaker_topic_relations = {}
    edges = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            fields = line.strip().split('\t')
            statement = fields[2] if len(fields) > 2 else None
            speaker = fields[4] if len(fields) > 4 else None
            label = fields[1].lower() if len(fields) > 1 else None  # Converti il label in lowercase

            if speaker:
                speakers.add(speaker)

            if len(fields) > 3:
                topic_list = fields[3].split(',')
                for topic in topic_list:
                    topics.add(topic.strip())

            if speaker and label:
                for topic in topic_list:
                    edge_key = f"{speaker}-{topic.strip()}"
                    speaker_topic_relations[edge_key] = categorize_label(label)
                    edges[(speaker, topic.strip())] = categorize_label(label)

    #print("Edges", edges)
    return speakers, topics, edges, speaker_topic_relations
'''


def count_news_occurrences_and_weight_by_label(file_path):
    speaker_label_topic_occurrences = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            fields = line.strip().split('\t')
            speaker = fields[4] if len(fields) > 4 else None
            label = fields[1].lower() if len(fields) > 1 else None  # Converti il label in lowercase
            topics = fields[3].split(',') if len(fields) > 3 else []

            if speaker and label:
                for topic in topics:
                    topic = topic.strip()
                    if speaker not in speaker_label_topic_occurrences:
                        speaker_label_topic_occurrences[speaker] = {}

                    if topic not in speaker_label_topic_occurrences[speaker]:
                        speaker_label_topic_occurrences[speaker][topic] = {'true': 0, 'mostly-true': 0, 'half-true': 0,
                                                                            'barely-true': 0, 'false': 0, 'pants-fire': 0,
                                                                            'total_weight': 0}

                    # Incrementa il conteggio per il label corrispondente
                    speaker_label_topic_occurrences[speaker][topic][label] += 1

                    # Calcola il peso come il prodotto delle occorrenze e del valore numerico del label
                    label_value = categorize_label(label)
                    weight = speaker_label_topic_occurrences[speaker][topic].get('total_weight', 0)
                    speaker_label_topic_occurrences[speaker][topic]['total_weight'] = weight + label_value

    return speaker_label_topic_occurrences


def print_stats_and_create_CSV_file(speaker_label_topic_occurrences):
    # Crea una lista di tuple (speaker, topic, total_weight)
    stats_list = []
    for speaker, topic_info in speaker_label_topic_occurrences.items():
        for topic, label_info in topic_info.items():
            total_weight = label_info.get('total_weight', 0)
            stats_list.append((speaker, topic, total_weight))

    # Ordina la lista in base al Total Weight decrescente
    sorted_stats = sorted(stats_list, key=lambda x: x[2], reverse=True)

    # Genera il nome del file CSV con la data e l'ora corrente
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_filename = f"Results_{current_datetime}.csv"

    # Stampare la tripla Speaker-Topic-Total Weight ordinata e scrivere nel file CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Speakers', 'Topics', 'Total Weights'])
        for speaker, topic, total_weight in sorted_stats:
            print(f"Speaker: {speaker}, Topic: {topic}, Total Weight: {total_weight}")
            csv_writer.writerow([speaker, topic, total_weight])




def build_bipartite_graph_with_weights(file_path, speaker_label_topic_occurrences):
    # Estrai gli speaker e i topic
    speakers, topics = set(), set()
    for speaker, label_info in speaker_label_topic_occurrences.items():
        speakers.add(speaker)
        topics.update(label_info.keys())
    topics.discard('total_weight')

    # Crea un grafo bipartito
    G = nx.Graph()

    # Aggiungi i nodi per gli speaker e i topic
    G.add_nodes_from(speakers, bipartite=0)
    G.add_nodes_from(topics, bipartite=1)

    # Aggiungi gli archi tra gli speaker e i topic con i pesi forniti
    for speaker, label_info in speaker_label_topic_occurrences.items():
        for topic, info in label_info.items():
            if topic != 'total_weight':
                weight = info['total_weight']
                edge_color = 'blue' if weight == 0 else 'gray'
                #edge_color = 'gray'
                edge_width = 1 if weight == 0 else weight * 2
                #edge_width = weight * 2
                G.add_edge(speaker, topic, weight=weight, color=edge_color, width=edge_width)

    # Disegna il grafo bipartito
    pos = nx.bipartite_layout(G, speakers)

    # Disegna i nodi
    nx.draw_networkx_nodes(G, pos, nodelist=speakers, node_color='red', label="Speakers")
    nx.draw_networkx_nodes(G, pos, nodelist=topics, node_color='green', label="Topics")

    # Disegna gli archi
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], width=edge[2]['width'], edge_color=edge[2]['color'])

    plt.legend()

    # Aggiungi etichette agli archi
    #edge_labels = {(edge[0], edge[1]): edge[2]['weight'] for edge in G.edges(data=True) if edge[2]['weight'] != 0}
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    # Aggiungi le etichette ai nodi
    labels = {}
    for node in G.nodes():
        labels[node] = node

    nx.draw_networkx_labels(G, pos, labels, font_size=10)



    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    file_path = 'dataLIAR/test.tsv'
    speaker_label_topic_occurrences = count_news_occurrences_and_weight_by_label(file_path)
    print_stats_and_create_CSV_file(speaker_label_topic_occurrences)
    build_bipartite_graph_with_weights(file_path, speaker_label_topic_occurrences)

