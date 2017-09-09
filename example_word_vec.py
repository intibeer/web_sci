#!/usr/bin/python

import csv 
import sys, os, math, time, argparse, shutil, gzip
import numpy as np
from math import log
from math import *
import spacy
import nltk
from numpy import dot 
from numpy.linalg import norm
from numpy import dot 
from numpy.linalg import norm 
from spacy.en import English
parser = English()

#Generate word vector of the word - apple  
apple = parser.vocab[u'school']

#Cosine similarity function 
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
others = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != unicode("apple")})

# sort by similarity score
others.sort(key=lambda w: cosine(w.vector, apple.vector)) 
others.reverse()

word_sim = [[cosine(w.vector, apple.vector), w.orth_]for w in others[:10]]

print "top most similar words to school:" 
for word in others[:10]:
    print word.orth_

print(word_sim)

for w in word_sim:
	print(str(w[1]) + " & " +  str(w[0]) )