import os
import pickle

import pandas as pd

from basic_model import get_classificaton_results_tpnf


def save_in_CSV(data_list, output_dir, file_name):
    # Crea il percorso completo del file CSV_complete_dataset
    csv_file_path = os.path.join(output_dir, file_name + ".csv")

    # Crea un DataFrame pandas dalla lista di dati
    df = pd.DataFrame(data_list)

    # Salva il DataFrame nel file CSV_complete_dataset
    df.to_csv(csv_file_path, index=False)

    print("Dati salvati con successo in:", csv_file_path)


def crea_tabella(cartella):
    # Ottieni il nome della cartella
    nome_cartella = os.path.basename(os.path.normpath(cartella))

    # Lista per memorizzare i dati
    dati = []

    # Scorri tutti i file nella cartella
    for file in os.listdir(cartella):
        # Controlla se il file termina con "metrics.txt" e non contiene "std"
        if file.endswith("metrics.txt") and "std" not in file:
            # Ottieni il percorso completo del file
            percorso_file = os.path.join(cartella, file)

            # Leggi il contenuto del file
            with open(percorso_file, 'r') as f:
                contenuto = f.read()

            # Estrai il nome del classificatore dal nome del file
            classificatore = file.replace("metrics.txt", "").strip('_')

            # Supponiamo che le metriche siano nel formato "metrica: valore" e separate da newline
            metriche = {}
            for linea in contenuto.splitlines():
                if ': ' in linea:
                    metrica, valore = linea.split(': ')
                    metriche[metrica] = valore

            # Aggiungi il classificatore e le metriche ai dati
            dati.append({"Classifier": classificatore, **metriche})

    # Crea un DataFrame dai dati
    df = pd.DataFrame(dati)

    # Salva il DataFrame in un file CSV
    csv_file = os.path.join(cartella, f"{nome_cartella}.csv")
    df.to_csv(csv_file, index=False)

    print(f"Tabella salvata in {csv_file}")

def create_all_tables(dir):
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        # Controlla se la sub-directory è una directory effettiva
        if os.path.isdir(subdir_path):
            # Applica la funzione alla sub-directory
            crea_tabella(subdir_path)

if __name__ == '__main__':
    #save_in_CSV("data/features", "data/CSV_complete_dataset")

    # Esempio di utilizzo
    #cartella = "data/metrics/politifact_metrics/micro_macro_struct_temp_linguistic"
    #crea_tabella(cartella)
    create_all_tables('data/metrics/politifact_metrics')
    create_all_tables('data/metrics/gossipcop_metrics')

