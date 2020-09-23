# ai4eu-competences
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
