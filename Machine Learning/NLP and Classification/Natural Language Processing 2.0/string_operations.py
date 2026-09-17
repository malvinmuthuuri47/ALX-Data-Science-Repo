"""
This module is meant to be the playground for string manipulation
when it comes to NLP and classification
"""

import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# nltk.download()

nltk.data.path.append("D:/ALX/Data Science/Machine Learning/NLP and Classification/Natural Language Processing 2.0/NLTK Downloads Folder")

def remove_punctuation(words: str):
    words = words.lower()
    return ''.join([x for x in words if x not in string.punctuation])

data = "Hi there. I like you and you are attractive to me. Your legs and complexion!" \
"Me likey :)"

res = remove_punctuation(data)
# print(res)

data = """
I really enjoyed the Python programming lessons. 
The programming exercises were challenging, but I learned a lot.

I enjoyed the Python exercises because the lessons were practical.
The challenging exercises helped me learn programming.

The lessons were interesting, but some programming exercises were difficult.
I learned a lot from the practical examples
"""

'''Stemming'''
stemmer = SnowballStemmer('english')

'''Tokenisation'''
tokeniser = TreebankWordTokenizer()
tokens = tokeniser.tokenize(data)
print("After Tokenization: ", tokens)

'''Lemmatization'''
lemmatizer = WordNetLemmatizer()

'''Stopwords'''
english_stopwords = stopwords.words('english')

tokens_less_stopwords = [word for word in tokens if word.lower() not in english_stopwords]

print("\nTokens minus Stopwords: ", tokens_less_stopwords)

def bag_of_words_count(words, word_dict={}):
    """
    Count how many times each word occurs.
    """
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1

    return word_dict

bag_of_words = bag_of_words_count(tokens_less_stopwords, {})

print(f"\nBag of words: {bag_of_words}")