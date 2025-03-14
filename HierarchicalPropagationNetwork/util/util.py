import datetime
import time

'''
La classe tweet_node rappresenta un nodo all'interno di una struttura di dati ad albero, relativo alle interazioni sui 
social media, come Twitter. Ogni nodo ha diverse proprietà, come tweet_id, text, created_time, user_name, user_id, 
news_id, node_type, botometer_score e sentiment. La classe fornisce anche metodi per aggiungere figli di tipo retweet e 
reply, impostare il tipo di nodo e il nodo genitore, nonché ottenere il contenuto del nodo.
Attributi:
    tweet_id: Identificativo univoco del tweet.
    text: Testo del tweet.
    created_time: Data e ora di creazione del tweet.
    user_name: Nome dell'utente che ha pubblicato il tweet.
    user_id: Identificativo dell'utente che ha pubblicato il tweet.
    news_id: Identificativo della notizia associata al tweet.
    retweet_children: Elenco dei nodi tweet che sono stati retweettati da questo tweet.
    reply_children: Elenco dei nodi tweet che hanno risposto a questo tweet.
    children: Insieme dei nodi tweet che sono figli di questo tweet.
    sentiment: Sentimento associato al tweet.
    parent_node: Nodo genitore di questo tweet nel grafo.
    node_type: Tipo del nodo tweet (ad esempio, retweet, risposta, etc.).
    botometer_score: Punteggio del bot ottenuto tramite il botometer per identificare i bot sui social media.
Funzioni:
    __init__(): Metodo per inizializzare un oggetto tweet_node con i relativi attributi.
    __eq__(): Metodo per confrontare due oggetti tweet_node in base all'identificativo del tweet.
    __hash__(): Metodo per ottenere l'hash dell'oggetto tweet_node basato sull'identificativo del tweet.
    set_node_type(): Metodo per impostare il tipo di nodo tweet.
    set_parent_node(): Metodo per impostare il nodo genitore di questo tweet.
    add_retweet_child(): Metodo per aggiungere un nodo tweet retweettato come figlio di questo tweet.
    add_reply_child(): Metodo per aggiungere un nodo tweet che ha risposto a questo tweet come figlio.
    get_contents(): Metodo per ottenere i contenuti principali del tweet.
'''
class tweet_node:

    def __init__(self, tweet_id, text = None, created_time = None, user_name = None, user_id = None, news_id = None, node_type = None, botometer_score = None, sentiment= None):
        self.tweet_id = tweet_id
        self.text = text
        self.created_time = created_time
        self.user_name = user_name
        self.user_id = user_id

        self.news_id = news_id

        self.retweet_children = []
        self.reply_children = []
        self.children = set()

        self.sentiment = sentiment
        self.parent_node = None
        self.node_type = node_type
        self.botometer_score = botometer_score

    def __eq__(self, other):
        return self.tweet_id == other.tweet_id

    def __hash__(self):
        return hash(self.tweet_id)

    def set_node_type(self, node_type):
        self.node_type = node_type

    def set_parent_node(self, parent_node):
        self.parent_node = parent_node

    def add_retweet_child(self, child_node):
        self.retweet_children.append(child_node)
        self.children.add(child_node)

    def add_reply_child(self, child_node):
        self.reply_children.append(child_node)
        self.children.add(child_node)

    def get_contents(self):
        return {"tweet_id": str(self.tweet_id),
                "text": self.text,
                "created_time": self.created_time,
                "user_name": self.user_name,
                "user_id": self.user_id,
                "news_id": self.news_id,
                "node_type": self.node_type
                }

'''
Questa funzione converte una stringa di data e ora nel formato utilizzato da Twitter in un oggetto datetime di Python.
'''
def twitter_datetime_str_to_object(date_str):
    time_struct = time.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
    date = datetime.datetime.fromtimestamp(time.mktime(time_struct))
    return int(date.timestamp())
