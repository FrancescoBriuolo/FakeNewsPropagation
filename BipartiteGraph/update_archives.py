'''
UPDATE ARCHIVES. Serve ad aggiornare due files csv esistenti facendo lo scraping da 'politifact.com'. Inserendo una parola chiave, il file csv che viene
creato è 'archive_with_keywords.csv' e contiene le notizie più recenti, relative alle prime n pagine per la ricerca con la parola chiave
inserita.
Questo csv poi può essre arricchito con il programma update_archives.py, inserendo una parola chiave.
Se non viene inserita una parola chiave (""), si popola il file 'archive.csv' con le affermazioni più recenti per le prime n pagine specificate.

NB: Author è il checker della notizia, non chi l'ha messa in giro per primo.

PS. se si vuole tornare più indietro nel tempo, usare un n maggiore (alla linea 264)

'''
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re

# Create lists to store the scraped data
authors = []
dates = []
statements = []
sources = []
targets = []
keywords_list = []

# Define the valid target values
valid_targets = ['true', 'mostly-true', 'half-true', 'mostly-false', 'false', 'pants-fire']

'''
Funzione per estrarre il valore della label a partire dall'immagine
'''
def extract_rating(image_url):
    # Controlla se l'URL contiene "meter-xxx"
    match = re.search(r'meter-([a-z-]+)', image_url)
    if match:
        return match.group(1)

    # Se non c'è corrispondenza, controlla se contiene "ruling_xxx"
    match = re.search(r'tom_ruling_pof', image_url)
    if match:
        return 'pants-fire'

    return "None"

# Create a function to scrape the site
def scrape_website(page_number, keywords, last_statement):
    page_num = str(page_number)  # Convert the page number to a string

    stop = False

    if keywords:
        query = '+'.join(keywords)
        URL = f'https://www.politifact.com/search/factcheck/?page={page_num}&q={query}'
    else:
        URL = 'https://www.politifact.com/factchecks/list/?page=' + page_num  # Append the page number to complete the URL

    print(URL)

    webpage = requests.get(URL)  # Make a request to the website
    soup = BeautifulSoup(webpage.text, 'html.parser')  # Parse the text from the website

    count_statements = 0

    if keywords:
        # Find all divs with the class 'c-textgroup__title'
        title_divs = soup.find_all('div', attrs={'class': 'c-textgroup__title'})  # Get the tag and its class
        for title_div in title_divs:
            # Find the <a> tag within the div
            a_tag = title_div.find_all('a')
            statement_text = a_tag[0].text.strip()
            if statement_text == last_statement:
                stop = True
                break
            count_statements += 1
            statements.append(statement_text)
            keywords_list.append(', '.join(keywords) if keywords else 'None')

        # Find all divs with the class 'c-textgroup__author'
        sources_divs = soup.find_all('div', attrs={'class': 'c-textgroup__author'})
        for source_div in sources_divs[:count_statements]:
            # Find the <a> tag within the div
            a_tag = source_div.find('a')
            if a_tag:
                source_text = a_tag.text.strip()  # Get the text from the <a> tag
                sources.append(source_text)

        # Find all divs with the class 'c-textgroup__meta'
        meta_divs = soup.find_all('div', attrs={'class': 'c-textgroup__meta'})
        for meta_div in meta_divs[:count_statements]:
            # Extract the text and split it
            meta_text = meta_div.text.strip()
            if 'By ' in meta_text and '•' in meta_text:
                # Split the text into author and date
                parts = meta_text.split('•')
                author_text = parts[0].replace('By ', '').strip()
                date_text = parts[1].strip()
                authors.append(author_text)
                dates.append(date_text)

        # Find all divs with the class 'm-result__media'
        target_divs = soup.find_all('div', attrs={'class': 'm-result__media'})
        for target_div in target_divs[:count_statements]:
            # Find the <img> tag within the div
            img_tag = target_div.find('img', class_='c-image__thumb')
            if img_tag:
                img_src = img_tag['src']
                rating = extract_rating(img_src)
                if rating in valid_targets:
                    targets.append(rating)
                else:
                    # Remove the last added items if rating is not valid
                    if statements:
                        statements.pop()
                    if authors:
                        authors.pop()
                    if dates:
                        dates.pop()
                    if sources:
                        sources.pop()
                    if keywords_list:
                        keywords_list.pop()
    else:
        # Get the tags and its class
        statement_footer = soup.find_all('footer', attrs={'class': 'm-statement__footer'})  # Get the tag and its class
        statement_quote = soup.find_all('div', attrs={'class': 'm-statement__quote'})  # Get the tag and its class
        statement_meta = soup.find_all('div', attrs={'class': 'm-statement__meta'})  # Get the tag and its class
        target = soup.find_all('div', attrs={'class': 'm-statement__meter'})  # Get the tag and its class

        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']

        # Loop through the div m-statement__quote to get the link
        for i in statement_quote:
            link2 = i.find_all('a')
            statement_text = link2[0].text.strip()
            if statement_text == last_statement:
                stop = True
                break
            count_statements += 1
            statements.append(statement_text)
            keywords_list.append(None)

        # Loop through the footer class m-statement__footer to get the date and author
        for i in statement_footer[:count_statements]:
            link1 = i.text.strip()
            name_and_date = link1.split()
            first_name = name_and_date[1]
            last_name = name_and_date[2]
            if name_and_date[4] in months:
                full_name = first_name + ' ' + last_name
                month = name_and_date[4]
                day = name_and_date[5]
                year = name_and_date[6]
            else:
                full_name = first_name + ' ' + last_name + ' ' + name_and_date[3]
                month = name_and_date[5]
                day = name_and_date[6]
                year = name_and_date[7]
            date = month + ' ' + day + ' ' + year
            dates.append(date)
            authors.append(full_name)

        # Loop through the div m-statement__meta to get the source
        for i in statement_meta[:count_statements]:
            link3 = i.find_all('a')  # Source
            source_text = link3[0].text.strip()
            sources.append(source_text)

        # Loop through the target or the div m-statement__meter to get the facts about the statement (True or False)
        for i in target[:count_statements]:
            fact = i.find('div', attrs={'class': 'c-image'}).find('img').get('alt')
            if fact in valid_targets:
                targets.append(fact)
            else:
                # Remove the last added items if rating is not valid
                if statements:
                    statements.pop()
                if authors:
                    authors.pop()
                if dates:
                    dates.pop()
                if sources:
                    sources.pop()
                if keywords_list:
                    keywords_list.pop()

    return stop

'''
Funzione per estrarre l'ultimo statement relativo alla ricerca effettuata (con o senza keywords)
'''
def extract_last_statement(df_archive, keywords):
    #Caso in cui viene effettuata una ricerca con keywords e il file csv di partenza non contiene record con keywords null
    if keywords and 'keywords' in df_archive.columns and not df_archive['keywords'].isnull().all():
        # Estrarre l'ultima dichiarazione relativa alle keywords inserite dall'archivio
        # Filtra l'archivio per la keyword specificata
        filtered_archive = df_archive[df_archive['keywords'].str.contains(keywords[0], case=False, na=False)]

        # Estrai l'ultima dichiarazione dal DataFrame filtrato
        if not filtered_archive.empty:
            last_statement = filtered_archive['statement'].iloc[0]
        else:
            last_statement = None  # O qualche valore di default se non ci sono dichiarazioni corrispondenti
    # Caso in cui viene effettuata una ricerca con keywords e il file csv di partenza contiene solo record con keywords null
    elif keywords and ('keywords' not in df_archive.columns or df_archive['keywords'].isnull().all()):
        last_statement = None
    # Caso in cui viene effettuata una ricerca senza keywords
    elif not keywords:
        # Controlla se ci sono record con la colonna 'keywords' nulla
        if 'keywords' in df_archive.columns:
            filtered_archive = df_archive[df_archive['keywords'].isna()]

            # Estrai l'ultima dichiarazione dal DataFrame filtrato
            if not filtered_archive.empty:
                last_statement = filtered_archive['statement'].iloc[0]
            else:
                last_statement = df_archive['statement'].iloc[0]  # Usare il valore più recente con .iloc[-1]
        else:
            last_statement = df_archive['statement'].iloc[0]
    return last_statement

'''
Funzione per salvare i risultati della ricerca effettuata nel file csv specificato
'''
def results_to_CSV(n, keywords, CSV_file):
    # Caricare l'archivio esistente
    df_archive = pd.read_csv(CSV_file)

    last_statement = extract_last_statement(df_archive, keywords)
    #print(f"LAST STATEMENT for {keywords}: {last_statement}")

    # Loop through 'n-1' webpages to update the archive
    for i in range(1, n):
        stop = scrape_website(i, keywords, last_statement)
        if stop:
            break

    # Creare un nuovo DataFrame con i dati ottenuti
    data = pd.DataFrame({
        'author': authors,
        'statement': statements,
        'source': sources,
        'date': dates,
        'target': targets,
        'keywords': keywords_list
    })

    # Unire i nuovi dati con l'archivio esistente, aggiungendo i nuovi dati in cima
    df_archive = pd.concat([data, df_archive], ignore_index=True)

    # Salvare l'archivio aggiornato nel file CSV
    df_archive.to_csv(CSV_file, index=False, sep=',')

    print(f"\n Success: file {CSV_file} updated!\n\n")


# Definire le parole chiave, il file CSV e il numero di pagine per estrarre i dati più recenti
keywords = []
keyword = input("Inserire keyword: ")
if keyword:
    keywords.append(keyword)

CSV_file = "archive_with_keywords.csv" if keywords else "archive.csv"
n = 6

results_to_CSV(n, keywords, CSV_file)
