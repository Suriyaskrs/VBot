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

# Ensure necessary NLTK data is available
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

warnings.filterwarnings('ignore')

# Initialize chatbot variables
sent_tokens = []
word_tokens = []
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def initialize_chatbot():
    global sent_tokens, word_tokens, stop_words

    # Load data from GitHub
    url = 'https://raw.githubusercontent.com/Suriyaskrs/VBot/main/VBot/VIT.txt'  # Fixed URL
    response = requests.get(url)

    if response.status_code == 200:
        raw = response.text.lower()
    else:
        raise Exception("Failed to load chatbot data. Check the file URL.")

    sent_tokens = [section.strip() for section in raw.split("###") if section.strip()]
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
    if score > 60:
        return best_match
    return None


# Generating response
def response(user_response):
    if not sent_tokens:  # Ensure chatbot is initialized only once
        initialize_chatbot()

    robo_response = ''

    # Compute similarity using TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])

    # Get best match
    idx = np.argmax(vals)
    best_score = np.max(vals)

    # Set a threshold
    threshold = 0.2

    if best_score < threshold:
        return "I am sorry! I don't understand you."

    robo_response = sent_tokens[idx]
    return robo_response


# Initialize chatbot before starting the Streamlit app
initialize_chatbot()
