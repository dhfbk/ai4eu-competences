import argparse
import pandas as pd
import fasttext
import scipy.spatial
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process skills')
parser.add_argument('inputFile', help='Input ESCO competences CSV file (usually skills_LANG.csv)')
parser.add_argument('fastTextModel', help='fastText model file')
parser.add_argument('--compFile', help='JSON competences file')
args = parser.parse_args()

print("Loading input file")
inputFile = args.inputFile
df = pd.read_csv(inputFile)
compFile = args.compFile

compSentences = None
if compFile != None:
    compSentences = []
    with open(compFile) as json_file:
        data = json.load(json_file)
        for r in data:
            s = r['title'].encode('latin-1', 'ignore').decode('utf-8', 'ignore')
            s = s.strip()
            sobj = {"sentence": s, "id": int(r['id'])}
            compSentences.append(sobj)

print("Loading fastText")
fastTextModel = args.fastTextModel
model = fasttext.load_model(fastTextModel)

print("Loading vectors")
vectors = {}
ids = {}
for index, row in df.iterrows():
    sentence = row['preferredLabel']
    v = model.get_sentence_vector(sentence)
    vectors[sentence] = v
    ids[sentence] = row['conceptUri']

if compSentences == None:
    while True:
        print("Inserisci una frase: ", end="")
        input1 = input()
        v = model.get_sentence_vector(input1)
        results = []
        for vector in vectors:
            sim = 1 - scipy.spatial.distance.cosine(v, vectors[vector])
            r = {}
            r['sentence'] = vector
            r['sim'] = sim
            results.append(r)

        results.sort(key=lambda x: x['sim'], reverse=True)

        print("---")
        print("Frase inserita:", input1)
        for i in range(5):
            print("Frase più vicina:", results[i])
        print("---")

else:
    out = []
    for j in tqdm(range(len(compSentences))):
        # print(compSentences[j])
        s = compSentences[j]['sentence']
        compId = compSentences[j]['id']
        v = model.get_sentence_vector(s)
        results = []
        for vector in vectors:
            sim = 1 - scipy.spatial.distance.cosine(v, vectors[vector])
            r = {}
            r['sentence'] = vector
            r['sim'] = sim
            r['url'] = ids[vector]
            results.append(r)

        results.sort(key=lambda x: x['sim'], reverse=True)
        o = {}
        o['sentence'] = s
        o['id'] = compId
        o['results'] = []
        for i in range(5):
            o['results'].append(results[i])

        out.append(o)

    with open(compFile + ".out", 'w') as outfile:
        json.dump(out, outfile, indent=4)
