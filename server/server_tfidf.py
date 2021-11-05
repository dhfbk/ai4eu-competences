from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import time

import argparse
import pandas as pd
import fasttext
import scipy.spatial
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import urllib
import requests
import math
from itertools import islice

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import numpy as np

# parser = argparse.ArgumentParser(description='Process skills')
# parser.add_argument('inputFile', help='Input ESCO skills CSV file (usually skills_LANG.csv)')
# parser.add_argument('broaderSkillFile', help='Input ESCO relations CSV file (usually broaderRelationsSkillPillar.csv)')
# parser.add_argument('fastTextModel', help='fastText model file')
# args = parser.parse_args()

args = {
    "inputFile": "../../v1.0.8/skills_it.csv",
    "inputGroupFile": "../../v1.0.8/skillGroups_it.csv",
    "broaderSkillFile": "../../v1.0.8/broaderRelationsSkillPillar.csv",
    "fastTextModel": "../../cc.it.300.bin",
    "port": 7208,
    "tint-url": "http://dh-server.fbk.eu:8013/tint",
    "pickle-name": "../../corpus-title.pickle",
    "pickle-name-2": "../../corpus-description.pickle",
    "modelName": 'dbmdz/bert-base-italian-xxl-cased',
    "pickle-name-bert": "../../vectors-esco.pickle"
}

# Tutorial: https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.YMCY0JMzbs0

logging.basicConfig(level=logging.INFO)
server_address = ('', args['port'])

def getBertVector(sentence):
    new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    new_tokens.to(device)
    mydict = {}
    mydict['input_ids'] = new_tokens['input_ids']
    mydict['attention_mask'] = new_tokens['attention_mask']
    outputs = esco_model(**mydict)
    embeddings = outputs.last_hidden_state
    attention_mask = mydict['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().cpu().numpy()
    return mean_pooled

def getVector(o, vectors):
    total = 0.0
    vecTotal = np.transpose(np.zeros(300))
    for word in o:
        if word in vectors:
            vecTotal += vectors[word]['weight'] * vectors[word]['vector']
            total += vectors[word]['weight']
    if total > 0:
        return vecTotal / total
    else:
        return None

def runTint(sentence_text):
    # requires args['tint-url']
    goodTokens = []
    myobj = {'text' : sentence_text.strip().lower()}
    x = requests.post(args['tint-url'], data = myobj)
    data = json.loads(x.text)
    goodTokens = []
    for sentence in data['sentences']:
        for token in sentence['tokens']:
            pos = token['pos']
            # if pos.startswith("A") or pos.startswith("B") or pos.startswith("S") or pos.startswith("V"):
            #     if pos != "BN":
            #         goodTokens.append(token['lemma'])
            if pos.startswith("A") or pos.startswith("S") or pos.startswith("V"):
                goodTokens.append(token['lemma'])
    return goodTokens

class S(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        mylist = json.loads(post_data.decode('utf-8'))

        numResults = 2 * int(mylist['numResults'])

        # default or zero
        multipliers = [1.0, 1.0, 1.0]

        if int(mylist['run_type']) == -2:
            multipliers = [1.4, 1.0, 0.6]
        if int(mylist['run_type']) == -1:
            multipliers = [1.2, 1.0, 0.8]
        if int(mylist['run_type']) == 1:
            multipliers = [1.0, 1.0, 1.0]
        if int(mylist['run_type']) == 2:
            multipliers = [1.0, 1.2, 1.4]
        normalText = mylist['text']
        text = normalText.lower()

        if "multiplier1" in mylist:
            multipliers[0] = float(mylist['multiplier1'])
        if "multiplier2" in mylist:
            multipliers[1] = float(mylist['multiplier2'])
        if "multiplier3" in mylist:
            multipliers[2] = float(mylist['multiplier3'])

        # print(multipliers)

        # results_sent = {}
        # v = model.get_sentence_vector(text)
        # for key in sentence_vectors:
        #     vector = sentence_vectors[key]
        #     sim = 1 - scipy.spatial.distance.cosine(v, vector)
        #     results_sent[key] = sim

        results_sent = {}
        tokens = runTint(text)
        v = getVector(tokens, word_vectors2)
        for key in sentence_tfidf_vectors2:
            vector = sentence_tfidf_vectors2[key]
            sim = 1 - scipy.spatial.distance.cosine(v, vector)
            results_sent[key] = sim

        results_tfidf = {}
        # tokens = runTint(text)
        v = getVector(tokens, word_vectors)
        if v is not None:
            for key in sentence_tfidf_vectors:
                vector = sentence_tfidf_vectors[key]
                sim = 1 - scipy.spatial.distance.cosine(v, vector)
                results_tfidf[key] = sim

        # mancano results per sent
        sortedResults_tfidf = dict(sorted(results_tfidf.items(), key=lambda item: item[1], reverse=True))
        sortedResults_sent = dict(sorted(results_sent.items(), key=lambda item: item[1], reverse=True))

        values_tfidf = {}
        for key, sim in islice(sortedResults_tfidf.items(), 0, numResults):
            for multiplier in multipliers:
                sentence = idToSentenceMap[key]
                if sentence not in values_tfidf:
                    values_tfidf[key] = 0
                values_tfidf[key] += float(multiplier) * sim
                try:
                    key = broaderMap[key]
                    sentence = idToSentenceMap[key]
                    sim = results_tfidf[key]
                except:
                    break

        values_sent = {}
        for key, sim in islice(sortedResults_sent.items(), 0, numResults):
            for multiplier in multipliers:
                sentence = idToSentenceMap[key]
                if sentence not in values_sent:
                    values_sent[key] = 0
                values_sent[key] += float(multiplier) * sim
                try:
                    key = broaderMap[key]
                    sentence = idToSentenceMap[key]
                    sim = results_sent[key]
                except:
                    break

        values_tfidf = {k: v for k, v in sorted(values_tfidf.items(), key=lambda item: item[1], reverse=True)}
        outValues_tfidf = []
        for key in values_tfidf:
            thisObj = {}
            thisObj['text'] = idToSentenceMap[key]
            thisObj['key'] = key
            thisObj['words'] = corpus[key]
            thisObj['sim'] = values_tfidf[key]
            outValues_tfidf.append(thisObj)

        values_sent = {k: v for k, v in sorted(values_sent.items(), key=lambda item: item[1], reverse=True)}
        outValues_sent = []
        for key in values_sent:
            thisObj = {}
            thisObj['text'] = idToSentenceMap[key]
            thisObj['key'] = key
            thisObj['sim'] = values_sent[key]
            outValues_sent.append(thisObj)

        outValues_bert = []
        mean_pooled = getBertVector(normalText)
        c = {}
        for k in esco_vectors:
            sim = cosine_similarity(mean_pooled, esco_vectors[k])
            c[k] = sim
        s = sorted(c.items(), key=lambda item: item[1], reverse=True)
        c = dict(s[:numResults])
        for k in c:
            thisObj = {}
            thisObj['text'] = esco_corpus[k]
            thisObj['key'] = k
            thisObj['sim'] = float(c[k][0][0])
            outValues_bert.append(thisObj)

        outData = {}
        outData['values_tfidf'] = outValues_tfidf
        outData['values_sent'] = outValues_sent
        outData['values_bert'] = outValues_bert
        outData['multipliers'] = multipliers
        outData['tokens'] = tokens

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        ret = bytes(json.dumps(outData), 'UTF-8')
        self.wfile.write(ret)

logging.info("Loading input file")
inputFile = args['inputFile']
df = pd.read_csv(inputFile)
inputGroupFile = args['inputGroupFile']
gdf = pd.read_csv(inputGroupFile)
broaderSkillFile = args['broaderSkillFile']
bdf = pd.read_csv(broaderSkillFile)

broaderMap = dict()
for index, row in bdf.iterrows():
    if row.conceptType != "KnowledgeSkillCompetence":
        continue
    broaderMap[row.conceptUri] = row.broaderUri


corpus = {}
if os.path.exists(args['pickle-name']):
    logging.info("Loading corpus file")
    with open(args['pickle-name'], 'rb') as handle:
        corpus = pickle.load(handle)
else:
    logging.info("Running Tint")
    for index, row in df.iterrows():
        sentence_text = row['preferredLabel'] #+ " " + row['description']
        goodTokens = runTint(sentence_text)
        corpus[row['conceptUri']] = goodTokens
    for index, row in gdf.iterrows():
        sentence_text = row['preferredLabel']
        if type(sentence_text) != str and math.isnan(sentence_text):
            continue
        goodTokens = runTint(sentence_text)
        corpus[row['conceptUri']] = goodTokens
    logging.info("Writing corpus file")
    with open(args['pickle-name'], 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

textOnlyCorpus = []
for key in corpus:
    textOnlyCorpus.append(" ".join(corpus[key]))
vectorizer = TfidfVectorizer(use_idf=True)
vectorizer.fit_transform(textOnlyCorpus)

corpus2 = {}
if os.path.exists(args['pickle-name-2']):
    logging.info("Loading corpus file (2)")
    with open(args['pickle-name-2'], 'rb') as handle:
        corpus2 = pickle.load(handle)
else:
    logging.info("Running Tint (2)")
    for index, row in df.iterrows():
        sentence_text = row['preferredLabel'] + "\n" + row['description']
        goodTokens = runTint(sentence_text)
        corpus2[row['conceptUri']] = goodTokens
    for index, row in gdf.iterrows():
        sentence_text = row['preferredLabel']
        if type(sentence_text) != str and math.isnan(sentence_text):
            continue
        goodTokens = runTint(sentence_text)
        corpus2[row['conceptUri']] = goodTokens
    logging.info("Writing corpus file (2)")
    with open(args['pickle-name-2'], 'wb') as handle:
        pickle.dump(corpus2, handle, protocol=pickle.HIGHEST_PROTOCOL)

textOnlyCorpus2 = []
for key in corpus2:
    textOnlyCorpus2.append(" ".join(corpus2[key]))
vectorizer2 = TfidfVectorizer(use_idf=True)
vectorizer2.fit_transform(textOnlyCorpus2)

logging.info("Loading BERT")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args["modelName"])
esco_model = AutoModel.from_pretrained(args["modelName"])
esco_model.to(device)

esco_corpus = {}

logging.info("Loading sentences")
for index, row in df.iterrows():
    sentence_text = row['preferredLabel'] #+ " " + row['description']
    esco_corpus[row['conceptUri']] = sentence_text
for index, row in gdf.iterrows():
    sentence_text = row['preferredLabel']
    if type(sentence_text) != str and math.isnan(sentence_text):
        continue
    esco_corpus[row['conceptUri']] = sentence_text

if os.path.exists(args['pickle-name-bert']):
    logging.info("Loading file")
    with open(args['pickle-name-bert'], 'rb') as handle:
        esco_vectors = pickle.load(handle)
else:
    esco_vectors = {}
    for k in tqdm.tqdm(esco_corpus.keys()):
        sentence = esco_corpus[k]
        mean_pooled = getBertVector(sentence)
        esco_vectors[k] = mean_pooled
    logging.info("Saving file")
    with open(args['pickle-name-bert'], 'wb') as handle:
        pickle.dump(esco_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

logging.info("Loading fastText")
fasttext.FastText.eprint = lambda x: None
fastTextModel = args['fastTextModel']
model = fasttext.load_model(fastTextModel)

logging.info("Extracting word vectors")
word_vectors = {}
df_idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names(), columns=["idf_weights"])
for index, row in df_idf.iterrows():
    if index in model:
        o = { "weight": row['idf_weights'], "vector": model[index] }
        word_vectors[index] = o

logging.info("Calculating TF-IDF sentence vectors")
sentence_tfidf_vectors = {}
for key in corpus:
    o = corpus[key]
    vec = getVector(o, word_vectors)
    if vec is not None:
        sentence_tfidf_vectors[key] = vec

logging.info("Extracting word vectors (2)")
word_vectors2 = {}
df_idf2 = pd.DataFrame(vectorizer2.idf_, index=vectorizer2.get_feature_names(), columns=["idf_weights"])
for index, row in df_idf2.iterrows():
    if index in model:
        o = { "weight": row['idf_weights'], "vector": model[index] }
        word_vectors2[index] = o

logging.info("Calculating TF-IDF sentence vectors (2)")
sentence_tfidf_vectors2 = {}
for key in corpus2:
    o = corpus2[key]
    vec = getVector(o, word_vectors2)
    if vec is not None:
        sentence_tfidf_vectors2[key] = vec

logging.info("Loading sentence vectors")
sentence_vectors = {}
idToSentenceMap = {}
for index, row in df.iterrows():
    sentence_text = row['preferredLabel'].lower()
    # sentence_text = row['preferredLabel'].lower() + " " + row['description'].lower().replace("\n", " ")
    key = row['conceptUri']
    idToSentenceMap[key] = sentence_text
    # idToSentenceMap[key] = row['preferredLabel'].lower()
    v = model.get_sentence_vector(sentence_text)
    sentence_vectors[key] = v
for index, row in gdf.iterrows():
    sentence_text = row['preferredLabel']
    key = row['conceptUri']
    if type(sentence_text) != str and math.isnan(sentence_text):
        continue
    sentence_text = sentence_text.lower()
    idToSentenceMap[key] = sentence_text
    v = model.get_sentence_vector(sentence_text)
    sentence_vectors[key] = v

httpd = HTTPServer(server_address, S)
logging.info('Starting httpd...\n')
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
httpd.server_close()
logging.info('Stopping httpd...\n')
