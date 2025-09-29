import pandas as pd
import nltk
from nltk.corpus import words   

filename = "./output_data.csv"

df = pd.read_csv(filename)

drop_index = []
for i, rows in df.iterrows():
    if not (isinstance(rows["review_text"], str)):
        drop_index.append(i)
    elif (len(rows["review_text"]) < 10):
        drop_index.append(i)


df = df.drop(index=drop_index)

df = df["review_text"]

df.to_csv("./output_data.csv", index=False)

nltk.download("words")
english_words = set(words.words())

def word_ratio(text):
    if pd.isna(text):
        return 0
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0
    valid_words = sum(token in english_words for token in tokens)
    return valid_words / len(tokens)

df["word_ratio"] = df["text"].apply(word_ratio)

df = df[df["word_ratio"] > 0.3].drop(columns=["word_ratio"])

df.to_csv("./output_data.csv", index=False)