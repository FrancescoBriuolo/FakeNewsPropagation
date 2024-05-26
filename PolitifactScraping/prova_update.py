# Import the dependencies
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
def scrape_website(page_number, keywords=None):
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
            if a_tag[0].text.strip() == last_statement:
                stop = True
                break
            count_statements += 1
            statements.append(a_tag[0].text.strip())
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
                targets.append(rating)
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
            count_statements += 1
            statements.append(link2[0].text.strip())
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
            targets.append(fact)

# Definire le parole chiave e il file CSV
keywords = ["Biden"]
CSV_file = "archive_with_keywords.csv"

# Caricare l'archivio esistente
df_archive = pd.read_csv(CSV_file)
print("DF attuale: ", df_archive)

# # Estrarre l'ultima dichiarazione dall'archivio
# last_statement = df_archive['statement'].iloc[0]

if keywords:
    # Estrarre l'ultima dichiarazione relativa alle keywords inserite dall'archivio
    # Filtra l'archivio per la keyword specificata
    filtered_archive = df_archive[df_archive['keywords'].str.contains(keywords[0], case=False, na=False)]

    # Estrai l'ultima dichiarazione dal DataFrame filtrato
    if not filtered_archive.empty:
        last_statement = filtered_archive['statement'].iloc[0]
    else:
        last_statement = None  # O qualche valore di default se non ci sono dichiarazioni corrispondenti
else:
    last_statement = df_archive['statement'].iloc[0]

print("LAST STATEMENT: ", last_statement)

# Scraping della prima pagina
#scrape_website(2, keywords)

for i in range(1, 2):
    scrape_website(i, keywords)

# Creare un nuovo DataFrame con i dati ottenuti
data = pd.DataFrame({
    'author': authors,
    'statement': statements,
    'source': sources,
    'date': dates,
    'target': targets,
    'keywords': keywords_list
})

print("Nuovi dati: ", data)

# Unire i nuovi dati con l'archivio esistente, aggiungendo i nuovi dati in cima
df_archive = pd.concat([data, df_archive], ignore_index=True)
print("DF ARCHIVE: ", df_archive)

# Salvare l'archivio aggiornato nel file CSV
df_archive.to_csv(CSV_file, index=False, sep=',')

print(f"\n Success: file {CSV_file} updated!\n\n")
