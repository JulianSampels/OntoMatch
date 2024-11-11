# Ontology Matching

Graph Search Algorithm Based Prompt Generation for Ontology Matching.

Note: The repository consists of source codes of the paper "Exploring  Prompt Generation Utilizing Graph Search Algorithms for Ontology Matching" presented at [EU Semantics 2024](https://2024-eu.semantics.cc/). The paper can be accessed through the following link: [ebooks.iospress.nl/doi/10.3233/SSW240003](https://ebooks.iospress.nl/doi/10.3233/SSW240003).
## Folder Hierarchy 
```bash
├── data
├── results
└── src
    ├── AlignmentFormat.py
    ├── accronyms.json
    ├── batch_loaders
    │   ├── alignment.py
    │   ├── ontology_parsing
    │   └── random_walk.py
    ├── config.json
    ├── configMatcher.json
    ├── configMatcherImport.py
    ├── globals.py
    ├── llms
    ├── maximum_bipartite_matching
    ├── prompt_template_generator
    ├── run_matcher.py
    ├── track.py
    ├── utilsODS.py
    └── verbalizer

```


## Requirements
use Python version >=3.10
```bash
$ pip install -r requirements.txt
```
on macOS please run
```bash
$ brew install cmake
$ python3 -m nltk.downloader stopwords
```
## Dataset Folder:
update dataset paths at ```src/config.json```
## Configuration:
adjust the pipeline tasks and algorithm configurations at ```src/configMatcher.json```

## How to run
```bash
$ git clone https://github.com/JulianSampels/OntoMatch.git
```

download this [file](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EbL1yTauXtpEqs4Izc97WNIBhumczrDGTNQb47uYGzXqsg?e=I9B5pR) and extract it to `src/verbalizer/graph2text/outputs/t5-base_13881/`
```python
$ cd src
$ python3 run_matcher.py

```

## Results
all results can be found at ```results/result_RDF```<br>
best results on the [OAEI Conference track](https://oaei.ontologymatching.org/2024/conference/index.html) are at ```results/result_RDF/conference/treePromptVersion0```

## T-test
We conducted t-test with significance level 0.1 to evaluate the significance of the prompt types and algorithms.
The results can be seen on [t-test](https://github.com/JulianSampels/OntoMatch/blob/master/src/significance-test/Significance-Test.ipynb) folder.
