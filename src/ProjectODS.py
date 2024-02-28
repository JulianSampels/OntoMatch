import utils
import os
import sys
sys.path.append('./verbalizer/Prompt_generator2/')
import configODSImport
from prompt_generator import generatePromptTemplates
from verbalizer.Prompt_generator2 import generatePrompt
from maximum_bipartite_matching import generateMaximumBipartiteMatching
from AlignmentFormat import serialize_mapping_to_file
from batch_loaders.ontology_parsing import preprocessing
from llms.llm_prompting import LLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from numpy import dot
from numpy.linalg import norm
from track import Track
from batch_loaders.random_walk import *
print('finished imports')

cachedLLMPredictions = {}
llm = None
modelSim = None
preprocessorSim = None
configODS = configODSImport.getConfigODS()

def candidate_concept_sim(concept1, concept2):
    """computes similarity score between concept1 and concept2

    Args:
        concept1 (str): concept 1 name to be matched
        concept2 (str): concept 2 name to be matched

    Returns:
        float: similarity score between concept1 and concept2
    """
    #>100x faster now
    global modelSim
    global preprocessorSim
    if modelSim == None:
        modelSim = SentenceTransformer("all-MiniLM-L6-v2")
    if preprocessorSim == None:
        preprocessorSim = preprocessing.PreprocessingPipeline()

    cleaned_concept1 = " ".join(preprocessorSim.process(concept1))
    cleaned_concept2 = " ".join(preprocessorSim.process(concept2))

    concept1_embedding = modelSim.encode(cleaned_concept1)
    concept2_embedding = modelSim.encode(cleaned_concept2)
    cos_sim = util.cos_sim(concept1_embedding, concept2_embedding)
    #print(f"cosine similarity between {concept1} and {concept2} is {cos_sim.item()}")
    return cos_sim.item()

def testSimScore(triples, key1, key2, threshold):
    for keyA, keyB, score in triples:
        if (key1 == keyA and key2 == keyB and (type(score) == type('String') or score >= threshold)):
            return True
    return False

def getNeighbors(nodeName, ontology: Ontology, neighborhoodRange):
    return [ontology.onto_name + '#' + neighbor for neighbor in getNeighborsName(nodeName, ontology, neighborhoodRange)]

def getNeighborsName(nodeName, ontology: Ontology, neighborhoodRange):
    Neighbors = ontology.get_parents(nodeName) + ontology.get_childs(nodeName) + ontology.get_equivalents(nodeName)
    if neighborhoodRange > 1:
        for neighbor in Neighbors.copy():
            Neighbors += getNeighborsName(neighbor, ontology, neighborhoodRange - 1)
    return list(set(Neighbors))

def getLLMPrediction(key1, key2, llmMatchedFilePath):
    global cachedLLMPredictions
    global llm
    global configODS
    promptsPath = '/'.join(llmMatchedFilePath.split('/')[-2:])
    promptsPath = configODS.get('promptsPath') + promptsPath
    cache = cachedLLMPredictions.get(promptsPath)
    if not cache: 
        cachedLLMPredictions[promptsPath] = utils.importFromJson(llmMatchedFilePath)
        cache = cachedLLMPredictions.get(promptsPath)
    promptKey = key1 + ';' + key2
    yesOrNo = cache.get(promptKey)
    if yesOrNo != None:
        return yesOrNo
    else:
        if not os.path.exists(promptsPath): 
            print('not found:', promptsPath)
            return 'no'
        promptDict = utils.importFromJson(promptsPath)
        prompt = promptDict.get(promptKey)
        if not prompt: return 'no'
        if not llm: llm = LLM()
        yesOrNo = llm.get_prediction(prompt)
        cache[promptKey] = yesOrNo
        utils.save_json(cache, llmMatchedFilePath)
    return yesOrNo

def main():
    """
    runs functions according to the tasks in configODS
    prepares configs, metrics for original project by FrancisGosselin

    Parameters:
    None

    Returns: 
    None
    """
    global configODS


    if (configODS.get('importOntologies') == True):
        #to stuff like in original project
        #get configFile from original project
        config = utils.importFromJson('config.json')
        #configure metric according to original project
        metrics_config={"results_files_path": "./result_alignments",
            "write_rdf": False,
            "write_tsv": False,
            "write_ranking": False,
            "hits":[1, 3, 5, 10], 
            "debug_files_path": "./debug"}
        
        #load ontologies
        name = "SmallTestCase"
        from transformers import AutoTokenizer
        Globals.tokenizer = AutoTokenizer.from_pretrained(config["General"]["model"])
        t = Track(name, config, metrics_config=metrics_config)
        ontos = {onto.get_name() : onto for onto in t.ontologies}


    #pre align all possible classes by cross products as candidates
    if (configODS.get('computeSimilaritiesExportAsAlignments') == True):
        #get which ontology maps to which one
        for ontoName1, ontoName2 in t.toBeMatchedOntologies:
            onto1 = ontos.get(ontoName1)
            onto2 = ontos.get(ontoName2)
            if onto1 and onto2:
                preAlignments = []
                for class1 in tqdm(onto1.get_classes(), desc = f'computing similarities for {ontoName1} X {ontoName2}'):
                    for class2 in onto2.get_classes():
                        similarity = candidate_concept_sim(class1, class2)
                        #if similarity > 0.2:
                        preAlignments.append([onto1.get_name() + '#' + class1, onto2.get_name() + '#' + class2, similarity])
                path = configODS.get('alignmentPath') + ontoName1 + '-' + ontoName2 + '.json'
                utils.saveToJson(preAlignments, path, messageText=f'exported preAlignments for ({ontoName1} X {ontoName2}) to ')

    #match exact matches
    if configODS.get('matchExactMatches'):
        for file_path in os.listdir(configODS.get('alignmentPath')):
            if file_path.endswith('.json'):
                alignmentFilePath = configODS.get('alignmentPath') + file_path
                triples = utils.importFromJson(alignmentFilePath)
                exactMatches = []
                for key1, key2, score in triples:
                    onto1, class1 = key1.split('#')
                    onto2, class2 = key2.split('#')
                    sim_core = score
                    if (sim_core >= 0.95):
                        alreadyIn = False
                        for classA, classB, _ in exactMatches:
                            if classA == class1 or classB == class2:
                                alreadyIn = True
                        if not alreadyIn: exactMatches.append([onto1 + '#' + class1, onto2 + '#' + class2, 1])
                #print(exactMatches)
                utils.saveToJson(exactMatches, configODS.get('exactMatchPath') + file_path, messageText='exported exact matches to')


    #run randomWalk Algorithm
    if configODS.get('runRandomWalkAlgorithm'):
        infer_walk_WALK = RandomWalkConfig(walk_type = 'randomWalk', strategy=WalkStrategy.ONTOLOGICAL_RELATIONS, n_branches=5)
        if ontos:
            for ontoName in ontos.keys():
                onto = ontos.get(ontoName)
                triples = {}
                for class1 in onto.get_classes():
                    triples.update(RandomWalk(onto, class1, infer_walk_WALK).triples)
                path = configODS.get('triplesPath') + "triples_randomWalk_" + onto.get_name() + '.json'
                utils.saveToJson(triples, path, f'exported random walk triples of {onto.get_name()} to')
    #run randomTree Algorithm
    if configODS.get('runRandomTreeAlgorithm'):
        randomTreeConfig = configODS.get('randomTreeConfig')
        if ontos and randomTreeConfig:
            for ontoName in ontos.keys():
                onto = ontos.get(ontoName)
                triples = {}
                for class1 in onto.get_classes():
                    tree = doRandomTree(onto, class1, randomTreeConfig)
                    if tree != None:
                        triplesOfClass1 = []
                        for classA in tree.keys():
                            for relation, classB in tree.get(classA):
                                classA = classA.replace('_', ' ')
                                relation = relation.replace('_', ' ')
                                classB = classB.replace('_', ' ')
                                triplesOfClass1.append([classA, relation, classB])
                        triples[onto.get_name() + '#' + class1] = triplesOfClass1
                path = configODS.get('triplesPath') + "triples_randomTree_" + onto.get_name() + '.json'
                utils.saveToJson(triples, path, f'exported random tree triples of {onto.get_name()} to')
    
    #export prompts generated by generatePromptTemplates.py with alignment candidates and context triples extracted by random algorithms
    #promptVersions: list of integers indicating which prompt versions to use
    promptVersion = configODS.get('promptsFoExportToJson')
    if promptVersion and (configODS.get('exportWalkPromptsToJson') or configODS.get('exportTreePromptsToJson')):
        print(f"promptVersions: {promptVersion}")
        for i in promptVersion:
            for file_path in os.listdir(configODS.get('alignmentPath')):
                #print(f"processing '{file_path}'")
                if file_path.endswith('.json'):
                    alignmentFilePath = configODS.get('alignmentPath') + file_path
                    if configODS.get('exportWalkPromptsToJson'):
                        if not os.path.exists(configODS.get('promptsPath') + f"walkPromptVersion{i}/"):
                            os.mkdir(configODS.get('promptsPath') + f"walkPromptVersion{i}/")
                        contextPaths = [configODS.get('triplesVerbalizedPath') + 'verbalized_triples_randomWalk_' + ontoName + '.json' for ontoName in file_path.replace('.json', '').split('-')]
                        promptList = generatePromptTemplates.getPrompt(alignmentFilePath, contextPaths, promptVersion = i, promptCounter = -1, skipIfNoContext = True)
                        utils.saveToJson(promptList, configODS.get('promptsPath') + f"walkPromptVersion{i}/"+ file_path, f"exported 'walkPromptVersion{i}' with alignments '{alignmentFilePath} and context '{contextPaths}' to")
                    if configODS.get('exportTreePromptsToJson'):
                        if not os.path.exists(configODS.get('promptsPath') + f"treePromptVersion{i}/"):
                            os.mkdir(configODS.get('promptsPath') + f"treePromptVersion{i}/")
                        contextPaths = [configODS.get('triplesVerbalizedPath') + 'verbalized_triples_randomTree_' + ontoName + '.json' for ontoName in file_path.replace('.json', '').split('-')]
                        promptList = generatePromptTemplates.getPrompt(alignmentFilePath, contextPaths, promptVersion = i, promptCounter = -1, skipIfNoContext = True)
                        utils.saveToJson(promptList, configODS.get('promptsPath') + f"treePromptVersion{i}/"+ file_path, f"exported 'treePromptVersion{i}' with alignments '{alignmentFilePath} and context '{contextPaths}' to")
    
    if configODS.get('verbalizeAvailableTriples'):
        for file_path in os.listdir(configODS.get('triplesPath')):
            if file_path.endswith('.json'):
                tripleFilePath = configODS.get('triplesPath') + file_path
                tripleVerbalizedFilePath = configODS.get('triplesVerbalizedPath') + 'verbalized_' + file_path
                generatePrompt.verbaliseFile(tripleFilePath, tripleVerbalizedFilePath)

    if configODS.get('runPromptsOnLLM'):
        global llm
        if not llm: llm = LLM()
        for dir_path in os.listdir(configODS.get('promptsPath')):
            #print(f"processing '{file_path}'")
            for file_path in os.listdir(configODS.get('promptsPath') + dir_path):
                if file_path.endswith('.json'):
                    promptsPath = configODS.get('promptsPath') + dir_path + '/' + file_path
                    llmMatchedFilePath = configODS.get('llmMatchedPath') + dir_path
                    if not os.path.exists(llmMatchedFilePath):
                        os.mkdir(llmMatchedFilePath + '/')
                    llmMatchedFilePath += '/' + file_path
                    promptDict = utils.importFromJson(promptsPath)
                    promptResult = {}
                    for promptKey in tqdm(promptDict, desc=f'running prompts for {file_path.replace(".json", "")}'):
                        prompt = promptDict.get(promptKey)
                        yesOrNo = llm.get_prediction(prompt)
                        promptResult[promptKey] = yesOrNo
                    utils.saveToJson(promptResult, llmMatchedFilePath)
    if configODS.get('generateMaximumBipartiteMatching'):
        for dir_path in os.listdir(configODS.get('promptsPath')):
            if os.path.isdir(configODS.get('promptsPath') + '/' + dir_path):
                for file_path in tqdm(os.listdir(configODS.get('promptsPath') + dir_path), desc=f'matching {dir_path}'):
                    if file_path.endswith('.json'):
                        llmMatchedFilePath = configODS.get('llmMatchedPath') + dir_path + '/' + file_path
                        bipartiteMatchingPath = configODS.get('bipartiteMatchingPath') + dir_path
                        if not os.path.exists(bipartiteMatchingPath):
                            os.mkdir(bipartiteMatchingPath + '/')
                        bipartiteMatchingPath += '/' + file_path
                        llmMatchedClasses = utils.importFromJson(llmMatchedFilePath)

                        verticesL = set()
                        verticesR = set()
                        edges = {}
                        alreadyMatched = {}
                        exactMatches = utils.importFromJson(configODS.get('exactMatchPath') + file_path)
                        for key1, key2, _ in exactMatches:
                            alreadyMatched[key1] = key2     #prevent key1 from getting matched to anything else in ontology2
                            alreadyMatched[key2] = key1     #prevent key2 from getting matched from anything else in ontology1
                            edges[key1] = [key2]
                            verticesL.add(key1)
                            verticesR.add(key2)
                        newMatches = generateMaximumBipartiteMatching.findMaximumBipartiteMatching(list(verticesL), list(verticesR), edges)
                        ontoName1, node1 = key1.split('#')
                        ontoName2, node2 = key2.split('#')
                        onto1 = ontos.get(ontoName1)
                        onto2 = ontos.get(ontoName2)

                        matching = newMatches
                        
                        neighborhoodRange = configODS.get('neighborhoodRange')
                        alignmentFilePath = configODS.get('alignmentPath') + file_path
                        triples = utils.importFromJson(alignmentFilePath)
                        i = 0
                        while len(newMatches) > 0:
                            i += 1
                            newNeighborhoodMatches = []
                            for currentNeighborhoodRange in range(1, 1 + neighborhoodRange):#start by matching near neighborhood and farther extend wider search
                                verticesL = set()
                                verticesR = set()
                                edges = {}
                                possibleAlignments = []
                                for key1, key2 in newMatches:
                                    _, node1 = key1.split('#')
                                    _, node2 = key2.split('#')
                                    neighborsOf1 = getNeighbors(node1, onto1, currentNeighborhoodRange)
                                    neighborsOf2 = getNeighbors(node2, onto2, currentNeighborhoodRange)
                                    possibleAlignments += [(keyA, keyB) for keyA in neighborsOf1 for keyB in neighborsOf2 if testSimScore(triples, keyA, keyB, configODS.get('thresholdForConsideration'))]
                                #ask LLM for match or no match
                                for keyA, keyB in tqdm(possibleAlignments, desc = f'running prompts for {currentNeighborhoodRange}-hop neighborhoods in {ontoName1}-{ontoName2}'):
                                    if not alreadyMatched.get(keyA) and not alreadyMatched.get(keyB):
                                        #if llmMatchedClasses.get(keyA + ';' + keyB) == 'yes':
                                        if getLLMPrediction(keyA, keyB, llmMatchedFilePath) == 'yes':
                                            verticesL.add(keyA)
                                            verticesR.add(keyB)
                                            if not edges.get(keyA):
                                                edges[keyA] = []
                                            edges[keyA].append(keyB)
                                newNeighborhoodMatches = generateMaximumBipartiteMatching.findMaximumBipartiteMatching(list(verticesL), list(verticesR), edges)
                                matching += newNeighborhoodMatches
                                for keyX, keyY in newNeighborhoodMatches:
                                    alreadyMatched[keyX] = keyY
                                    alreadyMatched[keyY] = keyX
                                # if len(newNeighborhoodMatches) > 0:
                                #     print('matchRound:', i, 'currentNeighborhoodRange:', currentNeighborhoodRange, 'found', len(newNeighborhoodMatches), 'in poolsize:', len(possibleAlignments))
                            newMatches = newNeighborhoodMatches
                        #add remaining llmMatched alignments with enough cosimilarity
                        # alignmentFilePath = configODS.get('alignmentPath') + ontoName1 + '-' + ontoName2 + '.json'
                        # preAlignments = utils.importFromJson(alignmentFilePath)
                        # verticesL = set()
                        # verticesR = set()
                        # edges = {}
                        # for keyA, keyB, similarity in preAlignments:
                        #     if similarity > 0.75 and not alreadyMatched.get(keyA) and not alreadyMatched.get(keyB):
                        #         if llmMatchedClasses.get(keyA + ';' + keyB) == 'yes':
                        #             verticesL.add(keyA)
                        #             verticesR.add(keyB)
                        #             if not edges.get(keyA):
                        #                 edges[keyA] = []
                        #             edges[keyA].append(keyB)
                        # simMatches = generateMaximumBipartiteMatching.findMaximumBipartiteMatching(list(verticesL), list(verticesR), edges)
                        # matching += simMatches
                        # newMatches += simMatches
                        # for keyX, keyY in simMatches:
                        #     alreadyMatched[keyX] = keyY
                        #     alreadyMatched[keyY] = keyX
                        utils.saveToJson(matching, bipartiteMatchingPath)
    if configODS.get('exportFinalMatchingsToRDF'):
        for dir_path in os.listdir(configODS.get('bipartiteMatchingPath')):
            for file_path in os.listdir(configODS.get('bipartiteMatchingPath') + dir_path):
                if file_path.endswith('.json'):
                    bipartiteMatchingPath = configODS.get('bipartiteMatchingPath') + dir_path + '/' + file_path
                    rdfPath = configODS.get('rdfPath') + dir_path
                    if not os.path.exists(rdfPath):
                        os.mkdir(rdfPath + '/')
                    rdfPath += '/' + file_path.replace('.json', '') + '.rdf'
                    bipartiteMatchedClasses = utils.importFromJson(bipartiteMatchingPath)
                    t = [('http://' + key1, 'http://' + key2, '=', 1.0) for key1, key2 in bipartiteMatchedClasses]
                    serialize_mapping_to_file(rdfPath, t)
                    print('exported ' + rdfPath)




main()

#for preparing fake llmMatched file
# configODS = configODSImport.getConfigODS()
# path = configODS.get('llmMatchedPath')
# path += 'cmt-conference.json'
# print(path)
# data = utils.importFromJson(path)
# r = {}
# for key in data.keys():
#     r[key] = True if random.randint(0, 1) == 1 else False
# utils.saveToJson(r, path)