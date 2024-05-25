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

    if keywords:
        query = '+'.join(keywords)
        URL = f'https://www.politifact.com/search/factcheck/?page={page_num}&q={query}'
    else:
        URL = 'https://www.politifact.com/factchecks/list/?page=' + page_num  # Append the page number to complete the URL

    print(URL)

    webpage = requests.get(URL)  # Make a request to the website
    soup = BeautifulSoup(webpage.text, "html.parser")  # Parse the text from the website

    if keywords:
        # Find all divs with the class 'c-textgroup__title'
        title_divs = soup.find_all('div', attrs={'class': 'c-textgroup__title'})  # Get the tag and its class
        for title_div in title_divs:
            # Find the <a> tag within the div
            a_tag = title_div.find('a')
            if a_tag:
                title_text = a_tag.text.strip()  # Get the text from the <a> tag
                statements.append(title_text)
                keywords_list.append(', '.join(keywords) if keywords else 'None')

        # Find all divs with the class 'c-textgroup__author'
        sources_divs = soup.find_all('div', attrs={'class': 'c-textgroup__author'})
        for source_div in sources_divs:
            # Find the <a> tag within the div
            a_tag = source_div.find('a')
            if a_tag:
                source_text = a_tag.text.strip()  # Get the text from the <a> tag
                sources.append(source_text)

        # Find all divs with the class 'c-textgroup__meta'
        meta_divs = soup.find_all('div', attrs={'class': 'c-textgroup__meta'})
        for meta_div in meta_divs:
            # Extract the text and split it
            meta_text = meta_div.text.strip()
            if 'By ' in meta_text and '•' in meta_text:
                # Split the text into author and date
                parts = meta_text.split('•')
                author_text = parts[0].replace('By ', '').strip()
                date_text = parts[1].strip()
                #print(date_text)
                authors.append(author_text)
                dates.append(date_text)

        # Find all divs with the class 'm-result__media'
        target_divs = soup.find_all('div', attrs={'class': 'm-result__media'})
        #print(target_divs)
        for target_div in target_divs:
            # Find the <img> tag within the div
            img_tag = target_div.find('img', class_='c-image__thumb')
            if img_tag:
                img_src = img_tag['src']
                rating = extract_rating(img_src)
                targets.append(rating)


    else:

        statement_footer = soup.find_all('footer', attrs={'class': 'm-statement__footer'})  # Get the tag and its class
        statement_quote = soup.find_all('div', attrs={'class': 'm-statement__quote'})  # Get the tag and its class
        statement_meta = soup.find_all('div', attrs={'class': 'm-statement__meta'})  # Get the tag and its class
        target = soup.find_all('div', attrs={'class': 'm-statement__meter'})  # Get the tag and its class

        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

        # Loop through the footer class m-statement__footer to get the date and author
        for i, j, k, l in zip(statement_footer, statement_quote, statement_meta, target):
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

            link2 = j.find_all('a')
            statement_text = link2[0].text.strip()

            link3 = k.find_all('a')  # Source
            source_text = link3[0].text.strip()

            fact = l.find('div', attrs={'class': 'c-image'}).find('img').get('alt')

            dates.append(date)
            authors.append(full_name)
            statements.append(statement_text)
            sources.append(source_text)
            targets.append(fact)
            keywords_list.append(None)


    print(statements)
    print(authors)
    print(sources)
    print(dates)
    print(targets)
    print(keywords_list)

    print(len(statements))
    print(len(authors))
    print(len(sources))
    print(len(dates))
    print(len(targets))
    print(len(keywords_list))


# Loop through 'n-1' webpages to initialize the archive
n = 5
keywords = ["Ukraine"]  # Replace with actual keywords or leave empty for no keywords
if keywords:
    CSV_file = "archive_with_keywords.csv"
else:
    CSV_file = "archive.csv"
for i in range(1, n):
    scrape_website(i, keywords)

# Create a new dataFrame
data = pd.DataFrame(columns=['author', 'statement', 'source', 'date', 'target', 'keywords'])
data['author'] = authors
data['statement'] = statements
data['source'] = sources
data['date'] = dates
data['target'] = targets
data['keywords'] = keywords_list

# Save the data set
data.to_csv(CSV_file, index=False, sep=',')
print("\n Success: file archive.csv created and initialized!\n\n")