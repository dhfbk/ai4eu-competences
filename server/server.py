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
    "port": 7208
}

# multipliers = [1, 0.9, 0.8]

class S(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        mylist = json.loads(post_data.decode('utf-8'))

        # Parameters: text, numResults, mult
        # if 'numResults' in mylist:
        #     numResults = int(mylist['numResults'])
        # else:
        #     numResults = 10
        numResults = 20

        multipliers = mylist['mult']
        text = mylist['text'].lower()

        results = {}
        v = model.get_sentence_vector(text)
        for vector in vectors:
            sim = 1 - scipy.spatial.distance.cosine(v, vectors[vector])
            results[vector] = sim

        # results.sort(key=lambda x: x['sim'], reverse=True)
        sortedResults = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        values = {}
        for sentence, sim in islice(sortedResults.items(), 0, numResults):
            for multiplier in multipliers:
                if sentence not in values:
                    values[sentence] = 0
                values[sentence] += float(multiplier) * sim
                try:
                    sentence = idToSentenceMap[broaderMap[sentenceToIdMap[sentence]]]
                    sim = results[sentence]
                except:
                    break

        values = {k: v for k, v in sorted(values.items(), key=lambda item: item[1], reverse=True)}
        outValues = []
        for key in values:
            thisObj = {}
            thisObj['text'] = key
            thisObj['sim'] = values[key]
            outValues.append(thisObj)

        outData = {}
        outData['values'] = outValues

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        ret = bytes(json.dumps(outData), 'UTF-8')
        self.wfile.write(ret)

print("Loading input file")
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

print("Loading fastText")
fastTextModel = args['fastTextModel']
model = fasttext.load_model(fastTextModel)

print("Loading vectors")
vectors = {}
sentenceToIdMap = {}
idToSentenceMap = {}
for index, row in df.iterrows():
    sentence = row['preferredLabel']
    v = model.get_sentence_vector(sentence)
    vectors[sentence] = v
    sentenceToIdMap[sentence] = row['conceptUri']
    idToSentenceMap[row['conceptUri']] = sentence
for index, row in gdf.iterrows():
    sentence = row['preferredLabel']
    if type(sentence) != str and math.isnan(sentence):
        continue
    v = model.get_sentence_vector(sentence)
    vectors[sentence] = v
    sentenceToIdMap[sentence] = row['conceptUri']
    idToSentenceMap[row['conceptUri']] = sentence

print(idToSentenceMap['http://data.europa.eu/esco/skill/a06a5a3f-07a9-40b3-bb7b-d93a960b1be5'])

logging.basicConfig(level=logging.INFO)
server_address = ('', args['port'])
httpd = HTTPServer(server_address, S)
logging.info('Starting httpd...\n')
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
httpd.server_close()
logging.info('Stopping httpd...\n')
