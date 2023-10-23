import nltk# Import the NLTK library and download the 'punkt' dataset
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer# Import the PorterStemmer 
import numpy as np
stemmer=PorterStemmer()# Create an instance of the PorterStemmer


 # Tokenize the input sentence into a list of words
def tokenize(sentence):   
    return nltk.word_tokenize(sentence)


def stem(word):   
    return stemmer.stem(word.lower())# Stem the input word to its root form & lowercased


def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
     
    # Stem each word in the tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1.0

    return bag

