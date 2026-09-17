import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download()

nltk.data.path.append("D:/ALX/Data Science/Machine Learning/NLP and Classification/Natural Language Processing 2.0/NLTK Downloads Folder")

text = """
The machine learning course is interesting.
The lessons are practical, and the exercises are useful.
I think the course is one of the best courses for beginners.
"""

# Get NLTK's English stop words
stop_words = set(stopwords.words('english'))

# Tokenize the text
tokens = word_tokenize(text)

# count stop-word occurrences
stop_word_count = sum(
    1 for word in tokens if word.lower() in stop_words
)

print("Number of stop words:", stop_word_count)