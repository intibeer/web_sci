#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv 
import sys, os, math, time, argparse, shutil, gzip
import numpy as np
from math import log
from math import *
import spacy
import nltk
from numpy import dot 
from numpy.linalg import norm
#from spacy.en import English
#parser = English()
#nlp = spacy.load('en')

#setup_workpath(workspace=args.workspace)

def entity_similarity(ep_tokens, gold_responses):
    gold_responses_tokens = zip(*gold_responses)
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    all_entity_sim = []
    #clean ep_tokens
    for sent_ in gold_responses_tokens[0]:
        are_ents = None
        turn_index = gold_responses_tokens[0].index(sent_)
        if len(convert_to_scy_doc(sent_)) == 0:
            print("No Entities Found")
            are_ents = False
            entity_sim = 0
        else:
            print("Entities Found")
            are_ents = True
            gold_entities = convert_to_scy_doc(sent_)
            gold_entities = [e.string for e in gold_entities]
            
            #print(gold_entities)
            li_li = [most_similar_words(word_.decode('utf-8')) for word_ in gold_entities]
            li = [item for sublist in li_li for item in sublist]
            gold_entities = gold_entities + li
            gold_entities = str("".join(gold_entities))

            gold_sim = gold_entities.split(" ")
            gold_sim = [g.encode('utf-8') for g in gold_sim]

            gold_entities = gold_entities.decode('utf-8')
            gold_entities = nlp(gold_entities)
            entities_vec = gold_entities.vector

        #convert ep_tokens to vectors
        if are_ents == True:
            for ep_token in ep_tokens[turn_index]:
                ep_token = [token.decode('unicode_escape') for token in ep_token]
                ep_token = str(" ".join(ep_token))
                ep_token = ep_token.decode('utf-8')
                ep_token = nlp(ep_token)
                ep_tokens_vec = ep_token.vector

            #print("ENT VEC@@@@@@@@@@@@")
            #print(entities_vec)
            #print("TOKEN VEC@@@@@@@@@@@@")
            #print(ep_tokens_vec)
            #print("VEC SIM @@@@@@@@@@@@")
            #print(cosine(entities_vec, ep_tokens_vec))
            #print("@@@@@@@@@@@END@@@@@@@@@@@@@@@@@@@")

            entity_sim = cosine(entities_vec, ep_tokens_vec)
            all_entity_sim.append(entity_sim)

    if len(all_entity_sim) == 0:
        final_score = 0
    else:
        final_score = sum(all_entity_sim)/len(all_entity_sim)

    return final_score

def convert_to_scy_doc(input_format):
    #input_format = [token.decode('unicode_escape') for token in input_format]
    #print(" ".join(input_format))
    input_format = list(set(input_format))
    input_format = [i.title() for i in input_format]
    doc = nlp(" ".join(input_format).decode('utf-8'))
    entities = []
    for e in doc.ents:
        entities.append(e)
    return entities

def most_similar_words(word):
    sim_words = []
    word_string = word
    word = word.decode('utf-8')
    word = parser.vocab[word]

    #Cosine similarity function 
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    others = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != unicode(word_string)})

    # sort by similarity score
    others.sort(key=lambda w: cosine(w.vector, word.vector)) 
    others.reverse()

    #print("10 most similar words to: " + word_string)
    for word in others[:10]:
        sim_words.append(word.orth_)

    return sim_words

print("*********************")


#gold_tokens = (['hello', 'this', 'is', 'a', 'test', 'london', 'O2', 'james', 'ben'], "34 54 75775"), (['word', 'help', 'use', 'cat'], "34 655 78 23")
gold_tokens= [(['i', 'know', 'one', 'thing', 'browns', 'will', 'miss', 'playoffs', 'lol'], '8 65 55 159 3464 61 322 2222 76'), (['this', 'is', 'the', 'funnest', 'fucking', 'thing', 'i', 'have', 'ever', 'read'], '21 14 5 19062 312 159 8 29 180 231')]

print(type(gold_tokens))
print(type(gold_tokens[0]))
print(type(gold_tokens[0][0]))
print(type(gold_tokens))
sent_1 = [ b'blue', b'sky']
sent_2 = [ b'i', b'really', b'really', b'like',b'bananas']
sent_3 = [ b'stupid', b'white', b'boy']
sent_4 = [ b'silly', b'brown', b'child']

ep_tokens = [sent_1, sent_2]

#print(entity_similarity(ep_tokens, gold_tokens))


#apples, and_, oranges = nlp(u'apples and oranges')


#Generate word vector of the word - apple  

#most_similar_words("house")

def calc_similarity(sent_1, sent_2):
    
    input_format_1 = [token.decode('unicode_escape') for token in sent_1]
    document_1 = nlp(" ".join(input_format_1))

    input_format_2 = [token.decode('unicode_escape') for token in sent_2]
    document_2 = nlp(" ".join(input_format_2))

    v1 = document_1.vector
    v2 = document_2.vector

    return v1.similarity(v2)

def lookup_ids(tf_ids, k, workspace):
    """

    Args: 
      tf_ids -> list of token ids 
      k -> number of converational steps
    Returns: 
      tuple, tokens and token IDs

    """
    print(os.path.dirname(os.path.realpath(__file__)))

    chat_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'works','lstm-baseline', 'data', 'chat.ids100000.in'))
    vocab_path_ = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'works','lstm-baseline', 'data', 'vocab100000.in'))
    #returns /Users/etz/thesis/jujube/models/k-thread-LSTM/lib
    #data_dir = "%s/data" % (workspace)
    #chat_path =  str(sys.path[-1]) + "/" + data_dir + "/chat.ids100000.in"
    #vocab_path_ = str(sys.path[-1]) + "/" + data_dir + "/vocab100000.in"

    with open(chat_path,'rb') as tk:
        sentences = tk.read()
        sentences = sentences.decode().split("\n")
        tk.close()
    tf_ids = " ".join([str(i) for i in tf_ids])
    sent_index = 0
    conv_indexes = []
    k_sents = []
    for s in sentences:
        if s == tf_ids:
          sent_index = sentences.index(s)
    conv_indexes.append(sent_index)
    for v in list(range(1, k)):
        sent_index += 2
        conv_indexes.append(sent_index)
    for ind in conv_indexes:
      k_sents.append(sentences[ind])
    with open(vocab_path_,'rb') as vo:
        tok = vo.read()
        tok = tok.decode().split("\n")
        vo.close()
        k_new = []
    for k_sen in k_sents:
      k_new.append([int(tt) for tt in k_sen.split(" ")])
    tokens_list = []
    for kn in k_new:
      tokens = []
      tokens = [tok[t] for t in kn]
      tokens_list.append(tokens)

    return list(zip(tokens_list, k_sents))


tf_ids = [41, 308, 15, 5, 414]


print(lookup_ids(tf_ids, 10, "..."))
"""

def get_entities(document):
    labels = set([w.label_ for w in document.ents]) 
    for label in labels: 
        entities = [e.string for e in document.ents if label==e.label_] 
        entities = list(set(entities)) 
        print(label, entities)

input_format_1 = [token.decode('unicode_escape') for token in sent_1]
#document_1 = nlp(" ".join(input_format_1).decode('utf-8'))



input_format_2 = [token.decode('unicode_escape') for token in sent_2]
document_2 = nlp(" ".join(input_format_2))

v1 = document_1.vector
v2 = document_2.vector




#doc = nlp(u'This is a sentence.')
#print(doc)
#apples, and_, oranges = nlp(u'shoe and lace')
#print(apples.vector.shape)
#print(apples.similarity(oranges))
#apple, and_, oranges = nlp(u'apple and orange')

#Cosine similarity function 
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
others = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != unicode("apple")})

#print(cosine(oranges.vector, apple.vector))
# sort by similarity score
#others.sort(key=lambda w: cosine(help.vector, apple.vector)) 
#others.reverse()


##print("top most similar words to apple:")
#for word in others[:10]:
#    print(word.orth_)


"""


"""
data = [[1076, 31, 2168, 3781, 7, 463, 16, 1154, 4, 37, 341, 129, 73, 28, 1447, 4], [129, 12, 11, 247, 28, 1604, 2064, 34, 93, 1175, 67447, 2456, 2]]
encoder, decoder = list(data[0]), list(data[1])
encoder = np.asarray(encoder, dtype=np.int) 
decoder = np.asarray(decoder, dtype=np.int)

#print("sizes")
#print(encoder.size)
#print(decoder.size)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sim_r(encoder, decoder):
    maxx = max([encoder.size, decoder.size])
    print(maxx)
    if encoder.size == decoder.size:
        pass
    elif encoder.size == maxx:
        print("padding decoder")
        padd = maxx - decoder.size
        decoder = np.concatenate((decoder, np.zeros((padd,), dtype=np.int)), axis =0)
        print(decoder.shape)
        print(encoder.shape)
        
    elif decoder.size == maxx:
        print("padding encoder")
        padd = maxx - encoder.size
        encoder = np.append(encoder, np.zeros((padd,), dtype=np.int))

    #encoder.append([0 for v in range(maxx - len(encoder))])
    
    #vec_a, vec_b = encoder, decoder

    #print(vec_a, vec_b)
    #r2 = sum(vec_a*vec_b) / sum(abs(vec_a)*abs(vec_b))
    #print(r2)
    #r2 = -log(abs(r2))
    #print(r2)
    #r2 = sigmoid(r2)
    #print(r2)
    #R = r2
    return encoder, decoder

#encoder, decoder = sim_r(encoder, decoder)
#print(nltk.f_measure(set(decoder.tolist()), set(encoder.tolist())))


******************************************************************************
************************      CODE DUMP     **********************************
******************************************************************************

#from tensorflow.python.ops import control_flow_ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spacy
import numpy
import csv
from config import params_setup
args = params_setup()

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
 
print(cosine_similarity(vec[0], vec[1]))

#example response
#resp_txt = [b'sounds', b'good', b'!', b'do', b'you', b'cook', b'too', b'?', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD']

resp_txt = [b'reinstalled', b'ffxiv', b'and', b'i', b'cannot', b'for', b'the', b'life', b'of', b'me', b'understand', b'why', b'_UNK', b'is', b'so', b'uncommon', b',', b'it', b"'", b's', b'awesome', b'.', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD', b'_PAD']
#load data -> returns list of strings.
def get_gold(workspace=args.workspace):
    data_dir = "%s/data" % (workspace)
    full_path =  str(sys.path[-1]) + "/" + data_dir + "/train/chat.txt.gz"
    with gzip.open(full_path,'rb') as zi:
        test_sentences = zi.read()
        test_sentences = test_sentences.decode("utf-8")
        test_sentences = test_sentences.split("\n")
        zi.close()
    return test_sentences

data = get_gold()

#example of data
print("********************")
print("example-> text: " + data[345])
print(type(data[345]))
print(len(data))
print("********************")

def reward(response, data):

    def response_get(response, data):
        ""
        args: a system respsonse plus data
        returns: a string of the gold response
        ""

        response = [r.decode("utf-8") for r in response]

        while u"_PAD" in response: response.remove(u"_PAD")
        symbols = ['?', '.', '!']
        for sym in symbols:
            for tok in response:
                if tok in sym:
                    response[response.index(tok)-1] = response[response.index(tok)-1] + response[response.index(tok)]
                    response.remove(tok)

        response = " ".join(response)
        qu_id = data.index(response)
        g_id = qu_id + 1 

        print(qu_id, g_id)
        print("........")
        print(data[345], data[346])
        return data[g_id], response 

    def calc_similarity(sent_1, sent_2):

        #document_1 = nlp(sent_1.decode('unicode_escape'))
        document_2 = nlp(sent_2)
        
        #input_format_1 = [token.decode('unicode_escape') for token in sent_1]
        document_1 = nlp(sent_1)


        #input_format_2 = [token.decode('unicode_escape') for token in sent_2]
        #document_2 = nlp(" ".join(input_format_2))
        #document_2 = nlp(" ".join(input_format_2))

        v1 = document_1.vector
        v2 = document_2.vector

        cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

        return cosine(v1, v2)

    query = response_get(response, data)
    print(query[0], query[1])
    return calc_similarity(query[1], query[0])

print(reward(resp_txt, data))

def setup_workpath(workspace):
  for p in ['data', 'nn_models', 'results']:
    wp = "%s/%s" % (workspace, p)
    if not os.path.exists(wp): os.mkdir(wp)

  data_dir = "%s/data" % (workspace)
  # training data

  if not os.path.exists("%s/chat.in" % data_dir):
    n = 0
    f_zip   = gzip.open("%s/train/chat.txt.gz" % data_dir, 'rt')
    f_train = open("%s/chat.in" % data_dir, 'w')
    f_dev   = open("%s/chat_test.in" % data_dir, 'w')
    for line in f_zip:
        print(line)

"""


