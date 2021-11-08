# Tool for matching competences

## Version 1
This tool computes the match between text and concepts from ESCO based on the ESCO model itself and the FastText computation model. Trustworthy is ensured in part by these models and their developers.

First of all, install the Python requirements running `pip install -r requirements.txt`.

Then, run `python skills.py`.

```
usage: skills.py [-h] [--compFile COMPFILE] inputFile fastTextModel

positional arguments:
  inputFile            Input ESCO competences CSV file (usually
                       skills_LANG.csv)
  fastTextModel        fastText model file

optional arguments:
  -h, --help           show this help message and exit
  --compFile COMPFILE  JSON competences file
```

The `fastTextModel` can be downloaded from the [FastText official page](https://fasttext.cc/).
The ESCO competences file is included in the [ESCO ontology](https://ec.europa.eu/esco/resources/data/static/model/html/model.xhtml) and can be downloaded from [here](https://ec.europa.eu/esco/portal).

If no other parameters are given to the script, a shell is opened and the user is asked to insert a row of text. After confirming by pressing enter, the system calculates the competence in the ESCO ontology that is semantically close to the input text.

If the `compFile` parameter is given, a new file is created, where each sentence in the input file is associated with the 5 semantically closest competences in ESCO. For each result, a similarity value is given. The output file is named `[compFile].out`.

The format of the `compFile` is:

```
[
   {
      "id":"1",
      "title":"A text to be compared with the ESCO ontology"
   },
   {
     ...
   }
 ]
 ```

## Version 2 (server)

Version 2 of the tool can be found in the `server` folder and can be run as a service, listening to a port.
The main script is `server_tfidf.py`, and one needs to fill the `args` variable at the beginning of the source file.

* `inputFile`: `skills_it.csv` file from ESCO ontology
* `inputGroupFile`: `skillGroups_it.csv` file from ESCO ontology
* `broaderSkillFile`: `broaderRelationsSkillPillar.csv` file from ESCO ontology
* `fastTextModel`: `cc.it.300.bin` file from FastText
* `port`: port where the server will listen to
* `tint-url`: URL of an instance of Tint, for example `http://localhost:8012/tint`, used for lemmatization and part-of-speech tagging
* `pickle-name`: output file for titles, for example `corpus-title.pickle` (it will speedup the execution)
* `pickle-name-2`: output file for descriptions, for example `corpus-description.pickle` (it will speedup the execution)
* `modelName`: BERT model (just leave `dbmdz/bert-base-italian-xxl-cased`)
* `pickle-name-bert`: output file for BERT vectors, for example `vectors-esco.pickle` (it will speedup the execution)

Model files can be downloaded on the respective websites.

* ESCO ontology is available on the [ESCO website](https://ec.europa.eu/esco/portal/home) (download the package from [here](https://ec.europa.eu/esco/portal/download))
* Tint can be downloaded on the [Tint website](https://tint.fbk.eu/) ([direct link](https://dhsite.fbk.eu/tint-release/0.3/tint-0.3-complete.tar.gz))
* FastText models can be downloaded from the [FastText models website](https://fasttext.cc/docs/en/crawl-vectors.html) ([direct link](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz))

Once the server is started, just call it on the corresponding port with at least the `text` argument.
