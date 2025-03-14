import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import datetime

'''
Converte la stringa "target" in un valore numerico corrispondente
'''
def categorize_target(target):
    if target == 'true':
        return 0
    elif target == 'mostly-true':
        return 0.2
    elif target == 'half-true':
        return 0.4
    elif target == 'mostly-false':
        return 0.6
    elif target == 'false':
        return 0.8
    elif target == 'pants-fire':
        return 1
    else:
        return None

'''
Legge il file CSV specificato e conta le occorrenze di notizie per ciascun "source" e "keyword", categorizzate per 
"target". Il dizionario "source_target_keyword_occurrences" tiene traccia del conteggio delle occorrenze di ciascun 
target e del peso totale. Il peso totale viene calcolato come la somma dei valori numerici dei target associati a 
ciascuna combinazione di source e keyword.
'''
def count_news_occurrences_and_weight_by_target(file_path):
    source_target_keyword_occurrences = {}

    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)  # Salta l'intestazione
        for fields in reader:
            source = fields[2].strip() if len(fields) > 2 else None
            target = fields[4].lower().strip() if len(fields) > 4 else None  # Converti il target in lowercase
            keywords = fields[5].split(',') if len(fields) > 5 else []

            if source and target:
                for keyword in keywords:
                    keyword = keyword.strip()
                    if source not in source_target_keyword_occurrences:
                        source_target_keyword_occurrences[source] = {}

                    if keyword not in source_target_keyword_occurrences[source]:
                        source_target_keyword_occurrences[source][keyword] = {
                            'true': 0,
                            'mostly-true': 0,
                            'half-true': 0,
                            'mostly-false': 0,
                            'false': 0,
                            'pants-fire': 0,
                            'total_weight': 0
                        }

                    # Incrementa il conteggio per il target corrispondente
                    source_target_keyword_occurrences[source][keyword][target] += 1

                    # Calcola il peso come il prodotto delle occorrenze e del valore numerico del target
                    target_value = categorize_target(target)
                    weight = source_target_keyword_occurrences[source][keyword].get('total_weight', 0)
                    source_target_keyword_occurrences[source][keyword]['total_weight'] = round(weight + target_value, 3)

    return source_target_keyword_occurrences

# Funzione per contare le occorrenze delle notizie e calcolare i pesi normalizzati
def count_news_occurrences_and_weight_by_target_normalized(file_path):
    normalized_source_target_keyword_occurrences = {}
    counts_news_source = {}

    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)  # Salta l'intestazione
        for fields in reader:
            source = fields[2].strip() if len(fields) > 2 else None
            target = fields[4].lower().strip() if len(fields) > 4 else None
            keywords = fields[5].split(',') if len(fields) > 5 else []

            if source and target:
                for keyword in keywords:
                    keyword = keyword.strip()
                    if source not in normalized_source_target_keyword_occurrences:
                        normalized_source_target_keyword_occurrences[source] = {}
                        counts_news_source[source] = {}

                    if keyword not in normalized_source_target_keyword_occurrences[source]:
                        normalized_source_target_keyword_occurrences[source][keyword] = {
                            'true': 0,
                            'mostly-true': 0,
                            'half-true': 0,
                            'mostly-false': 0,
                            'false': 0,
                            'pants-fire': 0,
                            'total_weight': 0,
                            'news_count': 0
                        }
                        counts_news_source[source][keyword] = 0

                    normalized_source_target_keyword_occurrences[source][keyword][target] += 1
                    normalized_source_target_keyword_occurrences[source][keyword]['news_count'] += 1
                    counts_news_source[source][keyword] += 1

                    target_value = categorize_target(target)
                    weight = normalized_source_target_keyword_occurrences[source][keyword]['total_weight']
                    normalized_source_target_keyword_occurrences[source][keyword]['total_weight'] = weight + target_value

    for source, keywords_info in normalized_source_target_keyword_occurrences.items():
        for keyword, target_info in keywords_info.items():
            if target_info['news_count'] > 0:
                print(source, keyword)
                print(target_info['news_count'])
                target_info['total_weight'] = round(target_info['total_weight'] / target_info['news_count'], 3)

    return normalized_source_target_keyword_occurrences, counts_news_source


def print_stats_and_create_CSV_file(source_target_keyword_occurrences, extension_name=''):
    # Crea una lista di tuple (source, keyword, total_weight)
    stats_list = []
    for source, keyword_info in source_target_keyword_occurrences.items():
        for keyword, target_info in keyword_info.items():
            total_weight = target_info.get('total_weight', 0)
            stats_list.append((source, keyword, total_weight))

    # Ordina la lista in base al Total Weight decrescente
    sorted_stats = sorted(stats_list, key=lambda x: x[2], reverse=True)

    # Genera il nome del file CSV con la data e l'ora corrente
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_filename = f"{extension_name}results_{current_datetime}.csv"

    # Stampare la tripla Source-Keyword-Total Weight ordinata e scrivere nel file CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sources', 'Keywords', 'Total Weights'])
        for source, keyword, total_weight in sorted_stats:
            print(f"Source: {source}, Keyword: {keyword}, Total Weight: {total_weight}")
            csv_writer.writerow([source, keyword, total_weight])


def build_bipartite_graph_with_weights(file_path, source_target_keyword_occurrences):
    # Estrai gli source e i keyword
    sources, keywords = set(), set()
    for source, target_info in source_target_keyword_occurrences.items():
        sources.add(source)
        keywords.update(target_info.keys())
    keywords.discard('total_weight')

    # Crea un grafo bipartito
    G = nx.Graph()

    # Aggiungi i nodi per gli source e i keyword
    G.add_nodes_from(sources, bipartite=0)
    G.add_nodes_from(keywords, bipartite=1)

    # Aggiungi gli archi tra gli source e i keyword con i pesi forniti
    for source, target_info in source_target_keyword_occurrences.items():
        for keyword, info in target_info.items():
            if keyword != 'total_weight':
                weight = info['total_weight']
                edge_color = 'blue' if weight == 0 else 'gray'
                edge_width = 1 if weight == 0 else weight * 2
                G.add_edge(source, keyword, weight=weight, color=edge_color, width=edge_width)

    # Disegna il grafo bipartito
    pos = nx.bipartite_layout(G, sources)

    # Disegna i nodi
    nx.draw_networkx_nodes(G, pos, nodelist=sources, node_color='red', label="Sources")
    nx.draw_networkx_nodes(G, pos, nodelist=keywords, node_color='green', label="Keywords")

    # Disegna gli archi
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], width=edge[2]['width'], edge_color=edge[2]['color'])

    plt.legend()

    # Aggiungi le etichette ai nodi
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.axis('off')
    plt.show()


def plot_weights_by_keyword(source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in source_target_keyword_occurrences.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        weights = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in source_target_keyword_occurrences.items():
            if keyword in keywords_info:
                sources.append(source)
                weights.append(round(keywords_info[keyword]['total_weight'], 3))

        # Crea un DataFrame per la keyword corrente
        df = pd.DataFrame({'source': sources, 'weight': weights})
        df = df.sort_values(by='weight', ascending=False)  # Ordina per peso in ordine decrescente

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(df['source'], df['weight'], color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Weight')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}.png')
        plt.close()

def plot_counts_and_weights_by_keyword(counts_news_source, normalized_source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        news_counts = []
        weights = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                news_counts.append(keywords_info[keyword])
                weights.append(normalized_source_target_keyword_occurrences[source][keyword]['total_weight'])

        # Ordina le source in base al numero di notizie pubblicate
        sorted_indices = sorted(range(len(news_counts)), key=lambda i: news_counts[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(sorted_sources, sorted_weights, color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Normalized Total Weight')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}_counts_weights.png')
        plt.close()

def plot_news_counts_by_keyword(counts_news_source, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        news_counts = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                news_counts.append(keywords_info[keyword])

        # Ordina le source in base al numero di notizie pubblicate
        sorted_indices = sorted(range(len(news_counts)), key=lambda i: news_counts[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_news_counts = [news_counts[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(sorted_sources, sorted_news_counts, color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Number of News')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}_news_counts.png')
        plt.close()

def plot_sorted_counts_and_weights_by_keyword(counts_news_source, normalized_source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        weights = []
        news_counts = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                weights.append(normalized_source_target_keyword_occurrences[source][keyword]['total_weight'])
                news_counts.append(keywords_info[keyword])

        # Ordina le source in base al peso totale normalizzato
        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(sorted_sources, sorted_weights, color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Normalized Total Weight')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}sorted_barplot_{keyword}_weights.png')
        plt.close()

def plot_counts_by_keyword_weight_ordered(counts_news_source, normalized_source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        weights = []
        news_counts = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                weights.append(normalized_source_target_keyword_occurrences[source][keyword]['total_weight'])
                news_counts.append(keywords_info[keyword])

        # Ordina le source in base al peso totale normalizzato
        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_news_counts = [news_counts[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(sorted_sources, sorted_news_counts, color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Number of News')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}_news_counts_weight_ordered.png')
        plt.close()

def plot_non_normalized_weights_by_keyword(counts_news_source, source_target_keyword_occurrences, normalized_source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        normalized_weights = []
        non_normalized_weights = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                normalized_weights.append(normalized_source_target_keyword_occurrences[source][keyword]['total_weight'])
                non_normalized_weights.append(source_target_keyword_occurrences[source][keyword]['total_weight'])

        # Ordina le sources in base al peso totale normalizzato
        sorted_indices = sorted(range(len(normalized_weights)), key=lambda i: normalized_weights[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_non_normalized_weights = [non_normalized_weights[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))
        plt.bar(sorted_sources, sorted_non_normalized_weights, color='blue')
        plt.xlabel('Sources')
        plt.ylabel('Non-normalized Total Weight')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}_non_normalized_weights.png')
        plt.close()

def plot_all_weights_by_keyword(counts_news_source, source_target_keyword_occurrences, normalized_source_target_keyword_occurrences, extension_name=''):
    # Trova tutte le keyword uniche
    all_keywords = set()
    for target_info in counts_news_source.values():
        all_keywords.update(target_info.keys())

    # Loop attraverso ogni keyword
    for keyword in all_keywords:
        sources = []
        normalized_weights = []
        non_normalized_weights = []
        news_counts = []

        # Raccogli dati per la keyword corrente
        for source, keywords_info in counts_news_source.items():
            if keyword in keywords_info:
                sources.append(source)
                normalized_weights.append(normalized_source_target_keyword_occurrences[source][keyword]['total_weight'])
                non_normalized_weights.append(source_target_keyword_occurrences[source][keyword]['total_weight'])
                news_counts.append(keywords_info[keyword])

        # Ordina le sources in base al peso totale normalizzato
        sorted_indices = sorted(range(len(normalized_weights)), key=lambda i: normalized_weights[i], reverse=True)
        sorted_sources = [sources[i] for i in sorted_indices]
        sorted_normalized_weights = [normalized_weights[i] for i in sorted_indices]
        sorted_non_normalized_weights = [non_normalized_weights[i] for i in sorted_indices]
        sorted_news_counts = [news_counts[i] for i in sorted_indices]

        # Crea il barplot
        plt.figure(figsize=(12, 8))

        # Barre per il peso normalizzato (verde)
        plt.bar(sorted_sources, sorted_normalized_weights, color='green', label='Normalized Weight')

        # Barre per il peso non normalizzato (blu)
        plt.bar(sorted_sources, sorted_non_normalized_weights, color='blue', label='Non-normalized Weight')

        # Barre per il numero di notizie (rosso)
        plt.bar(sorted_sources, sorted_news_counts, color='red', label='News Count')

        plt.xlabel('Sources')
        plt.ylabel('Weights / News Count')
        plt.title(f'Keyword: {keyword}')
        plt.xticks(rotation=90, fontsize=10)  # Ruota le etichette sull'asse x di 90 gradi
        plt.legend()

        # Aggiungi un margine per fare spazio alle etichette
        plt.subplots_adjust(bottom=0.25)

        # Salva il grafico come file immagine
        plt.savefig(f'{extension_name}barplot_{keyword}_all_weights.png')
        plt.close()



if __name__ == '__main__':
    file_path = 'dataPolitifact/archive_with_keywords.csv'
    source_target_keyword_occurrences = count_news_occurrences_and_weight_by_target(file_path)
    print_stats_and_create_CSV_file(source_target_keyword_occurrences)
    #build_bipartite_graph_with_weights(file_path, source_target_keyword_occurrences)

    plot_weights_by_keyword(source_target_keyword_occurrences)
    print("************************")
    normalized_source_target_keyword_occurrences, counts_news_source = count_news_occurrences_and_weight_by_target_normalized(file_path)

    print_stats_and_create_CSV_file(normalized_source_target_keyword_occurrences, 'normalized_')
    #build_bipartite_graph_with_weights(file_path, normalized_source_target_keyword_occurrences)

    plot_weights_by_keyword(normalized_source_target_keyword_occurrences, 'normalized_')

    print("Counts of news by source and keyword:")
    for source, keywords in counts_news_source.items():
        for keyword, count in keywords.items():
            print(f"{source}, {keyword}, {count}")

    #plot_counts_and_weights_by_keyword(counts_news_source, normalized_source_target_keyword_occurrences, 'normalized_')
    #plot_news_counts_by_keyword(counts_news_source, 'news_counts_')
    #plot_counts_and_weights_by_keyword(counts_news_source, source_target_keyword_occurrences)
    #plot_sorted_counts_and_weights_by_keyword(counts_news_source, normalized_source_target_keyword_occurrences, 'normalized_')
    plot_counts_by_keyword_weight_ordered(counts_news_source, normalized_source_target_keyword_occurrences,
                                          'news_counts_weight_ordered_')
    plot_non_normalized_weights_by_keyword(counts_news_source, source_target_keyword_occurrences,
                                           normalized_source_target_keyword_occurrences, 'non_normalized_weights_')
    plot_all_weights_by_keyword(counts_news_source, source_target_keyword_occurrences,
                                normalized_source_target_keyword_occurrences, 'all_weights_')