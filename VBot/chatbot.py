import os
import random
import string
import warnings
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('popular', quiet=True)  # for downloading necessary packages

warnings.filterwarnings('ignore')

# Reading in the corpus
sent_tokens = []
word_tokens = []
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def initialize_chatbot():
    global sent_tokens, word_tokens, stop_words

    
    nltk_data_path = "/home/appuser/nltk_data"  # Adjust based on your environment
    os.makedirs(nltk_data_path, exist_ok=True)  # Ensure directory exists
    nltk.data.path.append(nltk_data_path)  # Add to NLTK search paths

    # Ensure 'punkt' is available before using it
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=nltk_data_path)
    

  
    # Load and process data
    #replace your file path accordingly
    # Load and process data from GitHub
    url = 'https://raw.githubusercontent.com/Suriyaskrs/VBot/main/VBot/VIT.txt' 
    response = requests.get(url)

    if response.status_code == 200:  # Ensure successful fetch
        raw = response.text.lower()  # Get file content
    else:
        raise Exception("Failed to load chatbot data. Check the file URL.")

    sent_tokens = raw.split("\n\n")
    word_tokens = nltk.word_tokenize(raw)

    stop_words.update(set(stopwords.words('english')))  # Load stopwords


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching for Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Levenshtein Distance Handling for Typos
def get_best_match(user_query):
    best_match, score, _ = process.extractOne(user_query, sent_tokens)
    if score > 60:  # Threshold for similarity
        return best_match
    return None


# Generating response
def response(user_response):
    robo_response = ''
    initialize_chatbot()
    # Compute similarity using TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens + [user_response])  # Ensure user input is last
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare user input with all knowledge base entries

    # Get best match
    idx = np.argmax(vals)  # Get index of the best matching entry
    best_score = np.max(vals)

    # Set a threshold (adjustable)
    threshold = 0.2  # Lower threshold means more tolerance for errors

    if best_score < threshold:  # If similarity is too low, return a default response
        return "I am sorry! I don't understand you."

    # Return best match
    robo_response = sent_tokens[idx]
    return robo_response

