#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import telebot
import numpy
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving
from nltk.corpus import words
import re
from rdflib.plugins.sparql.processor import SPARQLResult
import rdflib
from rdflib import *


# In[ ]:


def get_processed_text(input):
    url = 'https://localhost:7082/api/check-input'
    row_review = {'review': input}
    return requests.post(url, json = row_review, verify=False)


# In[ ]:


def is_valid_english_text(text, threshold=0.7):
    if len(text) == 0:
        return False
    
    # Tokenize the text into words
    word_tokens = nltk.word_tokenize(text)

    # Count the number of valid English words
    valid_word_count = sum(word in words.words() for word in word_tokens)

    # Calculate the ratio of valid words to total words
    ratio = valid_word_count / len(word_tokens)

    # Check if the ratio is above the threshold
    return ratio >= threshold


# In[ ]:


def make_prediction(model, text):
    review_arr = numpy.array([text])
    res = model(review_arr)
    return res


# In[ ]:


def find_review_type_in_kb(model_index):
    result_review_type = None
    for review in positivity_rules['Positivity Status']:
        min_max = positivity_rules['Positivity Status'][review]
        r = range(min_max[-1], min_max[0] + 1)
        if model_index in r:
            result_review_type = review
            print(result_review_type)

    return result_review_type


# In[ ]:


loaded_model = tf.keras.saving.load_model('novel-forecast.keras')
bot = telebot.TeleBot('6788157679:AAHgf9IHeLAuLzcx-Al0Dfg3FU11l4_Xvno')


# In[ ]:


# Knowledge base query
KB_path = "./Goodreads.n3"
KB_uri = "file://" + KB_path.replace("\\", "/")
g = rdflib.Graph()
result = g.parse(KB_path, format="n3")


# In[ ]:


query = """
SELECT ?classLabel ?subclassLabel ?maxValue ?minValue
WHERE {
  ?class rdf:type owl:Class ;
         rdfs:label ?classLabel .

  ?subclass rdf:type owl:Class ;
             rdfs:subClassOf ?class ;
             rdfs:label ?subclassLabel ;
             prop:hasMax ?maxValue ;
             prop:hasMin ?minValue .
}
"""


# In[ ]:


knowledge_rows = g.query(query)


# In[ ]:


# Create rule base out of the KB query
positivity_rules = {}
for row in knowledge_rows:
    class_label = str(row['classLabel'])
    subclass_label = str(row['subclassLabel'])
    max_value = int(re.search(r'URN:inds:(\d+)', str(row['maxValue'])).group(1))
    min_value = int(re.search(r'URN:inds:(\d+)', str(row['minValue'])).group(1))
    if class_label not in positivity_rules:
        positivity_rules[class_label] = {}
    positivity_rules[class_label][subclass_label] = [max_value, min_value]


# In[ ]:


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    global predicted_reviews_count
    preprocessed_text = get_processed_text(message.text).text
    if is_valid_english_text(preprocessed_text):
        print(preprocessed_text)
        prediction = make_prediction(loaded_model, preprocessed_text)
        print(prediction)
        prediction_array = prediction
        max_index = numpy.argmax(prediction_array)
        result_message = find_review_type_in_kb(max_index + 1)
        bot.reply_to(message,
                     f'Hello I am a book review telegram bot, my prediction is: {result_message}!')  # bot.reply_to(message, f'The prediction is: {prediction}')
    else:
        bot.reply_to(message,
                     'Hello I am a book review telegram bot, your input does not look like a consistent book review. Please try again!')  # bot.reply_to(message, f'The prediction is: {prediction}')

# Wrong types of users inputs
@bot.message_handler(content_types=['sticker', 'document', 'photo'])
def handle_sticker(message):
    bot.reply_to(message, 'Hello I am a book review telegram bot, I can only handle textual input. Please try again!')


# In[ ]:


# Run the telegram bot
bot.infinity_polling()