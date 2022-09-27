from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import torch
import numpy as np
model = SentenceTransformer('whaleloops/phrase-bert')
with open('vocab.txt', 'r') as infile:
    vocab = list(infile.readlines())
with open('PhraseBERT.txt', 'w') as f:
    for word in vocab:
        output = ""
        output += word.strip() + " "
        for x in model.encode(word):
            output += str(x) + " "
        output += "\n"
        f.write(output)
