import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from dateutil import parser
from datetime import datetime


data_path = r"data/duplicates.json"


def normalization(text):
    '''
    Normalize the string by:
    - converting to lowercase
    - standardizing numeric expression 
    - removing any punctuation  
    - removing extra space

    parameters:
    text (can be a str or none)

    returns:
    Return an empty string if there is no input or return a normalized string
    '''
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\$?(\d+)\s*billion', r'\1b', text)
    text = re.sub(r'\$?(\d+)\s*million', r'\1m', text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date(item):
    '''

    It parses the 'date_published' field into a datatime object

    parameters:
    items (dict) : Contains information about the article, 'date_published' which is releveant to here'

    returns: 
    Returns the date if it exists otherwise it returns the max date
    '''
    try:
        return parser.parse(item["date_published"])
    except:
        return datetime.max


def main():
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]

    texts = []
    for item in items:
        # getting the raw text from items for it to be normalized
        # I gave context text full the most priority followed by
        # context text and title.
        
        text = (
            item.get("content_text_full")
            or item.get("content_text")
            or item.get("title", "")
        )
        texts.append(normalization(text))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dbscan = DBSCAN(
        # I tried have different values for eps, range from 0.20 --> 0.05
        # I got almost the same result
        
        eps=0.12,           
        min_samples=2,
        metric="cosine"
    )
    labels = dbscan.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        
        # This loop iterates over article indices and their assigned DBSCAN labels.
        # Articles with the same label are grouped into the same cluster.

        # If the label is -1, the article is considered noise (no similar articles),
        # so it is placed in its own single-item cluster.
        
        if label == -1:
            clusters[f"unique_{idx}"] = [idx]
        else:
            clusters.setdefault(label, []).append(idx)

    clean_items = []
    for cluster_indices in clusters.values():
        
        # For each cluster, find the article with the earliest publication date
        # and keep only that article in the cleaned results.
        
        earliest = min(cluster_indices, key=lambda i: parse_date(items[i]))
        clean_items.append(items[earliest])

    with open("clean_feed.json", "w", encoding="utf-8") as f:
        json.dump({"items": clean_items}, f, indent=2)


if __name__ == "__main__":
    main()
