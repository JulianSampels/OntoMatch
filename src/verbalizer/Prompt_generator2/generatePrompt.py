import json
from tqdm import tqdm
from verbalisation_module import VerbModule

verb_module = None

def verbalise(tripleList, verbModule, getL):
    ans = "translate Graph to English: "
    #print(tripleList)
    #predicateSet = set([])
    for item in tripleList:
        try :
            ans += f"<H> {getL(item[0])} <R> {item[1]} <T> {getL(item[2])} "  
        except Exception as e:
            print(item, 'error in generate Prompt, will continue:', repr(e))

    return verbModule.verbalise(ans)


def verbaliseFile(FILENAME, outputFile, getL):
    #load module only once
    global verb_module
    if not verb_module:
        verb_module = VerbModule()
    results = {}
    with open(FILENAME, "r") as f:
        data = json.loads(f.read())
    i = 0
    print(f'start generating "{outputFile}"')
    #only for progressbar
    #for key, value in data.items():
    for key, value in tqdm(data.items(), desc="Verbalizing", unit="item"):
        #compute verbalization
        triples = value
        verbalised_text = verbalise(triples, verb_module, getL)
        results.update({key: verbalised_text})
    json_object = json.dumps(results, indent=4)
    with open(outputFile, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    #for this path run from cd src/verbalizer/Prompt_generator2
    FILENAME = "../../../results/result_triples/conference/triples_randomTree_cmt.json"
    outputFile = "hello1.json"
    verbaliseFile(FILENAME, outputFile)