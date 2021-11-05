import requests
import json
import pprint
import os
import tqdm

sentencesFile = "../../sentences.json"
outDir = "../../outStats"
stopAt = 12
pp = pprint.PrettyPrinter(indent=4)

print("K =", stopAt)
sentencesData = []
with open(sentencesFile, "r") as f:
    sentencesData = json.load(f)

r = requests.get("https://dh-server.fbk.eu/cds/api/", params = {"action": "stats"})
ret = json.loads(r.text)
goldResults = ret["results"]

goldData = {}
sums_for_r = {}
for sentence in goldResults:
    goldData[sentence] = {}
    records = goldResults[sentence]
    sum_for_r = 0.0
    for comp_id in records:
        record = records[comp_id]
        value = 0.0
        if record['sum'] > 2:
            if record['avg'] >= 0.75:
                value = 1.0
            elif record['avg'] >= 0.5:
                value = 0.5
            elif record['avg'] >= 0.25:
                value = -0.5
            else:
                value = -1.0
        else:
            if record['avg'] >= 0.5:
                value = 0.5
            else:
                value = -0.5
        goldData[sentence][comp_id] = value
    for comp_id in goldData[sentence]:
        sum_for_r += goldData[sentence][comp_id]
    sums_for_r[sentence] = sum_for_r

indexes = set()
p_results = {}
f_best = {}
r_best = {}
for sentence in goldResults:
    p_results[sentence] = {}
    f_best[sentence] = {}
    r_best[sentence] = {}
    fileName = os.path.join(outDir, sentence)
    data = {}
    with open(fileName, "r") as f:
        data = json.load(f)

    for d in data:
        multIndex = "-".join(str(round(x, 1)) for x in d["results"]["multipliers"])
        indexes.add(multIndex)
        t = "values_tfidf"
        k = []
        for record in d["results"][t]:
            k.append(record["key"])
        p_sum = 0.0
        this_f_best = 0.0
        this_r_best = 0.0
        for pbase in range(stopAt):
            ptot = 0.0
            for i in range(pbase + 1):
                if k[i] in goldData[sentence]:
                    ptot += goldData[sentence][k[i]]
            p = ptot / (pbase + 1)
            r = ptot / sums_for_r[sentence]
            f = 0.0
            if p + r > 0:
                f = (2 * p * r) / (p + r)
            if f > this_f_best:
                this_f_best = f
            if r > this_r_best:
                this_r_best = r
            p_sum += p
        p_avg = p_sum / stopAt
        p_results[sentence][multIndex] = p_avg
        f_best[sentence][multIndex] = this_f_best
        r_best[sentence][multIndex] = this_r_best

final_f_best = {}
final_r_best = {}
final_p_avg = {}
for i in indexes:
    ptot = 0.0
    ftot = 0.0
    rtot = 0.0
    num = 0
    for sentence in p_results:
        ptot += p_results[sentence][i]
        ftot += f_best[sentence][i]
        rtot += r_best[sentence][i]
        num += 1
    p_avg = ptot / num
    f_avg = ftot / num
    r_avg = rtot / num
    final_p_avg[i] = p_avg
    final_f_best[i] = f_avg
    final_r_best[i] = r_avg
    # print(i, p_avg)
# pp.pprint(p_results)

sorted_p_avg = sorted(final_p_avg, key=final_p_avg.get, reverse=True)
sorted_f_best = sorted(final_f_best, key=final_f_best.get, reverse=True)
sorted_r_best = sorted(final_r_best, key=final_r_best.get, reverse=True)

print("P")
print(sorted_p_avg[0], final_p_avg[sorted_p_avg[0]])
print("1.0-0.9-0.8", final_p_avg["1.0-0.9-0.8"])
print(sorted_f_best[0], final_p_avg[sorted_f_best[0]])
print(sorted_r_best[0], final_p_avg[sorted_r_best[0]])

print("R")
print(sorted_r_best[0], final_r_best[sorted_r_best[0]])
print("1.0-0.9-0.8", final_r_best["1.0-0.9-0.8"])
print(sorted_p_avg[0], final_r_best[sorted_p_avg[0]])
print(sorted_f_best[0], final_r_best[sorted_f_best[0]])

print("F1")
print(sorted_f_best[0], final_f_best[sorted_f_best[0]])
print("1.0-0.9-0.8", final_f_best["1.0-0.9-0.8"])
print(sorted_p_avg[0], final_f_best[sorted_p_avg[0]])
print(sorted_r_best[0], final_f_best[sorted_r_best[0]])

    # manage multipliers
    # also values_sent
    # k = []
    # for record in data[0]["results"]["values_tfidf"]:
    #     k.append(record["key"])
    # p_sum = 0.0
    # for pbase in range(10):
    #     ptot = 0.0
    #     for i in range(pbase + 1):
    #         if k[i] in goldData[sentence]:
    #             ptot += goldData[sentence][k[i]]
    #     p = ptot / (pbase + 1)
    #     p_sum += p
    #     # print(pbase + 1, p)

    # p_avg = p_sum / 10
    # print(sentence, p_avg)

    # for k1 in k:
    #     if k1 in goldData[sentence]:
    #         print(goldData[sentence][k1])
    #     else:
    #         print("unknown")

exit()

actions = []
for sentence in sentencesData:
    sid = sentence["id"]
    text = sentence["sentence"]
    outFile = os.path.join(outDir, sid)
    for m1 in range(6, 15):
        for m2 in range(6, 15):
            for m3 in range(6, 15):
                thisRecord = {}
                thisRecord["multipliers"] = [m1, m2, m3]
                thisRecord["text"] = text
                thisRecord["action"] = "parse"
                thisRecord["fileName"] = outFile
                actions.append(thisRecord)

    thisRecord = {}
    thisRecord["action"] = "save"
    thisRecord["fileName"] = outFile
    actions.append(thisRecord)

toSave = []
for action in tqdm.tqdm(actions, miniters=1):
    if os.path.exists(action["fileName"]):
        continue

    if action["action"] == "save":
        with open(action["fileName"], 'w') as fw:
            json.dump(toSave, fw)
        toSave = []

    elif action["action"] == "parse":
        data = {
            "run_type": 0,
            "text": action["text"],
            "numResults": 20,
            "multiplier1": action["multipliers"][0] * 0.1,
            "multiplier2": action["multipliers"][1] * 0.1,
            "multiplier3": action["multipliers"][2] * 0.1
        }
        r = requests.post('https://dh-server.fbk.eu/cds-api/', data = json.dumps(data))
        ret = json.loads(r.text)
        thisRecord = {}
        thisRecord["multipliers"] = [m1, m2, m3]
        thisRecord["results"] = ret
        toSave.append(thisRecord)

exit()


for sentence in sentencesData:
    sid = sentence["id"]

    outFile = os.path.join(outDir, sid)
    if os.path.exists(outFile):
        print("File", outFile, "exists, skipping...")
        continue

    toSave = []
    text = sentence["sentence"]
    for m1 in range(6, 15):
        for m2 in range(6, 15):
            for m3 in range(6, 15):
                print(sid, "---", m1, m2, m3, "---", text)
                data = {
                    "run_type": 0,
                    "text": text,
                    "numResults": 20,
                    "multiplier1": m1 * 0.1,
                    "multiplier2": m2 * 0.1,
                    "multiplier3": m3 * 0.1
                }
                r = requests.post('https://dh-server.fbk.eu/cds-api/', data = json.dumps(data))
                ret = json.loads(r.text)
                thisRecord = {}
                thisRecord["multipliers"] = [m1, m2, m3]
                thisRecord["results"] = ret
                toSave.append(thisRecord)

    print("Saving file", outFile)
    with open(outFile, 'w') as fw:
        json.dump(toSave, fw)


