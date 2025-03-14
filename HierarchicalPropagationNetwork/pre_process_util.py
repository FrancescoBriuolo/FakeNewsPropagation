import json

'''
Questa funzione prende il percorso del file JSON data_file come input. Apre il file e itera su ogni riga. Ogni riga 
contiene un articolo di notizie nel formato JSON. Utilizzando json.loads(line), converte la riga JSON in un dizionario 
Python rappresentante l'articolo di notizie. Utilizza yield per restituire ogni articolo uno alla volta, trasformando la 
funzione in un generatore. Questo permette di leggere e processare gli articoli uno alla volta, riducendo la quantità di 
memoria necessaria per gestire grandi insiemi di dati.
'''
def get_news_articles(data_file):
    with open(data_file) as file:
        data = json.load(file)
        yield data
        #for line in file:
            #yield json.loads(line)


data_file = "data/nx_network_data/politifact_fake/politifact13038.json"
for article in get_news_articles(data_file):
    print(article)

