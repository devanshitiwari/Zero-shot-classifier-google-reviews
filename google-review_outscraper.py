
from outscraper import ApiClient

#dont run every time

api_cliet = ApiClient(api_key='')
reviews_response = api_cliet.google_maps_business_reviews(
    'Search Term', limit=395, language='en')
    
import pandas as pd 
review_d = {}
review_d = (reviews_response['reviews_data'])
#print(review_d)
type(review_d)

import json

json_formatted_str = json.dumps(reviews_response, indent=2)
#pretty print json
#print("Main Fields in pretty json format")
#print(json_formatted_str)
#storing json string to python objects


df = pd.DataFrame(review_d)
df.head(10)


review_dataframe = df.loc[:,['autor_name','review_text','review_rating','review_datetime_utc']]

review_dataframe.head(10)

review_dataframe.to_csv('review_dataframe_google_reviews.csv', index=False)

    
    
#if we want to scrape data directly for a search item

from bs4 import BeautifulSoup
import requests
import time
import json

### if we were to do it comment by comment - depending on UCDH API access.
# def main():
#     try:
#         #filename = "reviews.txt"
#         #file = open(filename,"w")
#         headers = {'User-agent' :'Mozilla/5.0'}

#         URL = "https://www.google.com/search?q=search+term#lrd=0x809ad07cdaea8a03:0x7b80e362defd80ba,1,,," 
#         page = requests.get(URL, headers=headers)
#         doc = BeautifulSoup(page.content, 'html.parser')
#         #print(doc)
#         #from the page content above we look for the class that has the link to the item.
#         div = doc.find_all("div",class_="Jtu6Td")
#         #print(div)
#         items_price = doc.find('span',class_="review_snippet")
#         print(items_price)
#             #saving all urls with a line change.
#             #file.write(str(review)+'\n')
#         #file.close()
#     except Exception as ex:
#         print("Error:" + str(ex))
        

# if __name__ == '__main__':
# 	main()



#################################PROCESSING TEXT _ NLP#############################################

from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.data import Sentence
from flair.trainers import ModelTrainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.optim.adam import Adam
import re, nltk, spacy

from flair.embeddings import TransformerDocumentEmbeddings


import ftfy
import string
from torch.optim.adam import Adam
import re, nltk, spacy
import pandas as pd
import numpy as np
#import language_tool_python
from transformers import pipeline
from numpy import argmax

zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')



hypothesis_template = "This text is about {}."

thisdict = {
    '':['','','','',''],
    '':['','','','',''],
    '':['','','','','']
}

classes = ['','','']


d = []
predicted_class = ""
for i, row in df.iterrows():
    predicted_class = ""
    text = row['review_text']
    text = ftfy.fix_text(text)
    text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    #text = tool.correct(text)
    #topic = row['topic']
    results = zero_shot_classifier(text, classes, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES = results["scores"]
    CLASSES = results["labels"]
    BEST_INDEX = argmax(SCORES)
    predicted_class = CLASSES[BEST_INDEX]
    # Print sentence with predicted labels
    print("iter: "+text+" TOPIC:"+predicted_class)
    
    #for first topic
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub1 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
    #for second topic
    predicted_class = CLASSES[BEST_INDEX+1]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub2 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
    
    #for third topic
    predicted_class = CLASSES[BEST_INDEX+2]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub3 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
       
    #for fourth topic
    predicted_class = CLASSES[BEST_INDEX+3]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub4 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
        
    d.append(
            {
                'text': row['review_text'],
                'author_name': row['autor_name'],
                'review_rating': row['review_rating'],
                'review_datetime_utc': row['review_datetime_utc'],
                'predicted_class1': CLASSES[BEST_INDEX],
                'predicted_class2': CLASSES[BEST_INDEX+1],
                'predicted_class3': CLASSES[BEST_INDEX+2],
                'predicted_class4': CLASSES[BEST_INDEX+3],
                'predicted_sub1': predicted_class_sub1,
                'predicted_sub2': predicted_class_sub2,
                'predicted_sub3': predicted_class_sub3,
                'predicted_sub4': predicted_class_sub4
            }
        )


    output = pd.DataFrame(d)
    output.to_csv('topics-subtopics_google_reviews.csv', index=False)


        
## SENTIMENTS


def clean(raw):
    """ Remove hyperlinks and markup """
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    return result


classifier = TextClassifier.load('./model_result/final-model.pt')



#df = review_dataframe
#df = df.head(10)

df['text'] = df['text'].fillna('').apply(str)




d = []

for i, row in df.iterrows():
    document =  row['text']
    document = clean(document)
    sentence = Sentence(document)
    classifier.predict(sentence)
    print(document+"\n\n")
    print(sentence.labels)
    print("\n\n")
    d.append(
        {
            'Sentiment': sentence.labels
        }
    )

df['flair_sentiment'] = d
output = pd.DataFrame(df)
output.to_csv('sentiment_flair_google_reviews.csv', index=False)

#Clean up using REGEX

import re
for i, row in output_try.iterrows():
    document =  row['flair_sentiment']
    document = str(document)
    match_object =  re.search("\[([a-zA-z]+)", str(document))
    output_try['flair_sentiment'] = match_object.group(1)
    
output_try.to_csv('sentiment_topics_clean_google_reviews.csv', index=False)




## KEYPHRASES 

from sklearn.feature_extraction.text import CountVectorizer
comments = output['text']
#choose number of words in each phrase here
n_gram_range = (4, 4)
stop_words = "english"

phrase_column = []

for doc in comments:
    try:
        # Extract candidate words/phrases
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
        candidates = count.get_feature_names()
        phrase_column.append(candidates) 
    except ValueError:
        phrase_column.append('') 
        pass
        


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding_column = []
candidate_embeddings_column = []

for doc in comments:
    try:
        doc_embedding = model.encode([doc])
        doc_embedding_column.append(doc_embedding) 
    except ValueError:
        doc_embedding_column.append('') 
        pass


for candidates in phrase_column:
    try:
        candidate_embeddings = model.encode(candidates)
        candidate_embeddings_column.append(candidate_embeddings)
    except ValueError:
        candidate_embeddings_column.append('') 
        pass
        
from sklearn.metrics.pairwise import cosine_similarity

top_n = 3

distances_column = []
keywords_column = []

for doc_embedding, candidate_embeddings, candidates in zip(doc_embedding_column, candidate_embeddings_column, phrase_column):
    try:
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

        distances_column.append(distances)
        keywords_column.append(keywords)
    except ValueError:
        distances_column.append('')
        keywords_column.append('')
        pass
output.to_csv('Google_Reviews.csv', index=False)
   
        
        
        
        
        
        
        
        
