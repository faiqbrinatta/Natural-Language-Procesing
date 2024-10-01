import nltk


text = "Who would have thought that computer programs would be analyzing human sentiments"

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)
