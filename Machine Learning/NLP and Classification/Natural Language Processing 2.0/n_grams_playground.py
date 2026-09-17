"""
This module introduces the concept of vectorization and n_grams in Machine Learning
"""

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I enjoy machine learning",
    "I enjoy Python Programming",
    "Machine Learning is interesting"
]

# Create the vectorizer
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

# Display the words that CountVectorizer learned
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# Display the numerical representation
print("\nDocument-term matrix:")
print(X.toarray())