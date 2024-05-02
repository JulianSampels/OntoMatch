import utilsODS
import os
import sys
import configMatcherImport
from prompt_template_generator import generatePromptTemplates
sys.path.append('./verbalizer/')
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
configODS = configMatcherImport.getConfig()


def candidate_concept_sim(concept1, concept2):
    """
    computes similarity score between concept1 and concept2

    Args:
        concept1 (str): concept 1 name to be matched
        concept2 (str): concept 2 name to be matched

    Returns:
        float: similarity score between concept1 and concept2
    """
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

"""
collects all neighbors with a maximal distance of neighborhoodRange to nodeName

Args:
    nodeName (str): name of the center node in ontology
    ontology (Ontology): the ontology in which the neighbors are collected
    neighborhoodRange (int): maximal distance which neighbors can have to center node with name nodeName

Returns:
    list of keys [ontologyName#neighborName] for nodes in neighborhood
"""
def getNeighbors(nodeName, ontology: Ontology, neighborhoodRange):
    return [ontology.onto_name + '#' + neighbor for neighbor in getNeighborsName(nodeName, ontology, neighborhoodRange)]

def getNeighborsName(nodeName, ontology: Ontology, neighborhoodRange):
    Neighbors = ontology.get_parents(nodeName) + ontology.get_childs(nodeName) + ontology.get_equivalents(nodeName)
    if neighborhoodRange > 1:
        for neighbor in Neighbors.copy():
            Neighbors += getNeighborsName(neighbor, ontology, neighborhoodRange - 1)
    return list(set(Neighbors))

"""
runs prompt on the LLM if not already cached in file
saves result to cache file

Args:
    key1 (str): key of the first node
    key2 (str): key of the second node
    llmOutcomePath (str): file path to the cached prompt answers

Returns:
    str: the answer the LLM gave; most of the time it is 'yes' or 'no'
"""
def getLLMPrediction(key1, key2, llmOutcomePath):
    global cachedLLMPredictions
    global llm
    global configODS
    promptsPath = '/'.join(llmOutcomePath.split('/')[-2:])
    promptsPath = configODS.get('promptsPath') + promptsPath
    #test if cache is loaded
    cache = cachedLLMPredictions.get(promptsPath)
    if not cache: 
        #load and prepare cache variable
        if os.path.exists(llmOutcomePath):
            cachedLLMPredictions[promptsPath] = utilsODS.importFromJson(llmOutcomePath)
            cache = cachedLLMPredictions.get(promptsPath)
        else:
            cachedLLMPredictions[promptsPath] = {}
            cache = cachedLLMPredictions.get(promptsPath)
    #test if already in cache
    promptKey = key1 + ';' + key2
    yesOrNo = cache.get(promptKey)
    if yesOrNo != None:
        return yesOrNo
    else:
        #not found => run on LLM and save new cache
        if not os.path.exists(promptsPath): 
            print('not found:', promptsPath)
            return 'no'
        promptDict = utilsODS.importFromJson(promptsPath)
        prompt = promptDict.get(promptKey)
        if not prompt: return 'no'
        if not llm: llm = LLM()
        yesOrNo = llm.get_prediction(prompt)
        cache[promptKey] = yesOrNo
        utilsODS.saveToJson(cache, llmOutcomePath, doPrint=False)
    return yesOrNo

def main():
    """
    runs functions according to the tasks in configODS
    usual workflow: 
                    1. import the file from the last pipeline step
                    2. do the demanded operations
                    3. save result to new result file

    Parameters:
    None

    Returns: 
    None
    """
    global configODS

    #import the ontologies for later usage
    if (configODS.get('importOntologies') == True):
        #get configFile from original project
        config = utilsODS.importFromJson('config.json')
        #load ontologies
        name = configODS.get('track')
        from transformers import AutoTokenizer
        Globals.tokenizer = AutoTokenizer.from_pretrained(config["General"]["model"])
        t = Track(name, config, metrics_config=None)
        ontologies = {onto.get_name() : onto for onto in t.ontologies}

    #compute similarities
    if (configODS.get('computeSimilarities') == True):
        #get which ontology maps to which one
        for ontoName1, ontoName2 in t.toBeMatchedOntologies:
            onto1 = ontologies.get(ontoName1)
            onto2 = ontologies.get(ontoName2)
            if onto1 and onto2:
                similarityTriplesPath = {}
                for class1 in tqdm(onto1.get_classes(), desc = f'computing similarities for {ontoName1} X {ontoName2}'):
                    for class2 in onto2.get_classes():
                        similarity = candidate_concept_sim(class1, class2)
                        similarityTriplesPath[onto1.get_name() + '#' + class1 + ';' + onto2.get_name() + '#' + class2] = [onto1.get_name() + '#' + class1, onto2.get_name() + '#' + class2, similarity]
                path = configODS.get('similarityPath') + ontoName1 + '-' + ontoName2 + '.json'
                utilsODS.saveToJson(similarityTriplesPath, path, messageText=f'exported similarities for ({ontoName1} X {ontoName2}) to ')

    #match exact matches higher than a threshold
    if configODS.get('matchExactMatches'):
        for file_path in os.listdir(configODS.get('similarityPath')):
            if file_path.endswith('.json'):
                alignmentFilePath = configODS.get('similarityPath') + file_path
                triples = list(utilsODS.importFromJson(alignmentFilePath).values())
                exactMatches = []
                for key1, key2, score in triples:
                    onto1, class1 = key1.split('#')
                    onto2, class2 = key2.split('#')
                    sim_core = score
                    if (sim_core >= configODS.get('thresholdForExactMatches')):
                        alreadyIn = False
                        for classA, classB, _ in exactMatches:
                            if classA == class1 or classB == class2:
                                alreadyIn = True
                        if not alreadyIn: exactMatches.append([onto1 + '#' + class1, onto2 + '#' + class2, 1])
                utilsODS.saveToJson(exactMatches, configODS.get('exactMatchPath') + file_path, messageText='exported exact matches to')

    #run randomWalk Algorithm and save walk as list of triples
    if configODS.get('runRandomWalkAlgorithm'):
        infer_walk_WALK = RandomWalkConfig(walk_type = 'randomWalk', strategy=WalkStrategy.ONTOLOGICAL_RELATIONS, n_branches=5)
        if ontologies:
            for ontoName in ontologies.keys():
                onto = ontologies.get(ontoName)
                triples = {}
                for class1 in onto.get_classes():
                    triples.update(RandomWalk(onto, class1, infer_walk_WALK).triples)
                path = configODS.get('triplesPath') + "triples_randomWalk_" + onto.get_name() + '.json'
                utilsODS.saveToJson(triples, path, f'exported random walk triples of {onto.get_name()} to')

    #run randomTree Algorithm and save tree as list of triples
    if configODS.get('runRandomTreeAlgorithm'):
        randomTreeConfig = configODS.get('randomTreeConfig')
        if ontologies and randomTreeConfig:
            for ontoName in ontologies.keys():
                onto = ontologies.get(ontoName)
                triples = {}
                for class1 in onto.get_classes():
                    tree = doRandomTree(onto, class1, randomTreeConfig)
                    #do some nice formatting
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
                utilsODS.saveToJson(triples, path, f'exported random tree triples of {onto.get_name()} to')
    
    #verbalize available triples
    if configODS.get('verbalizeAvailableTriples'):
        from verbalizer import tripleVerbalizer
        for file_path in os.listdir(configODS.get('triplesPath')):
            if file_path.endswith('.json'):
                tripleFilePath = configODS.get('triplesPath') + file_path
                tripleVerbalizedFilePath = configODS.get('triplesVerbalizedPath') + 'verbalized_' + file_path
                tripleVerbalizer.verbaliseFile(tripleFilePath, tripleVerbalizedFilePath)
    
    #generate prompt strings for all triples using the verbalized context extracted by the random algorithms
    #promptVersions: list of integers indicating which prompt versions to use
    promptVersion = configODS.get('promptVersions')
    if promptVersion and (configODS.get('generateWalkPrompts') or configODS.get('generateTreePrompts')):
        print(f"promptVersions: {promptVersion}")
        for i in promptVersion:
            for file_path in os.listdir(configODS.get('similarityPath')):
                if file_path.endswith('.json'):
                    alignmentFilePath = configODS.get('similarityPath') + file_path
                    if configODS.get('generateWalkPrompts'):
                        contextPaths = [configODS.get('triplesVerbalizedPath') + '_' + ontoName + '.json' for ontoName in file_path.replace('.json', '').split('-')]
                        promptList = generatePromptTemplates.getPrompt(alignmentFilePath, contextPaths, promptVersion = i, promptCounter = -1, skipIfNoContext = True)
                        utilsODS.saveToJson(promptList, configODS.get('promptsPath') + f"walkPromptVersion{i}/"+ file_path, f"exported 'walkPromptVersion{i}' with alignments '{alignmentFilePath} and context '{contextPaths}' to")
                    if configODS.get('generateTreePrompts'):
                        contextPaths = [configODS.get('triplesVerbalizedPath') + 'triples_randomTree_' + ontoName + '.json' for ontoName in file_path.replace('.json', '').split('-')]
                        promptList = generatePromptTemplates.getPrompt(alignmentFilePath, contextPaths, promptVersion = i, promptCounter = -1, skipIfNoContext = True)
                        utilsODS.saveToJson(promptList, configODS.get('promptsPath') + f"treePromptVersion{i}/" + file_path, f"exported 'treePromptVersion{i}' with alignments '{alignmentFilePath} and context '{contextPaths}' to")

    #runs ALL available prompts on the LLM, does not use the caching function!
    #normally it is not necessary to run ALL prompts on the LLM!
    #to run prompts dynamically as needed use "runMissingPromptsOnDemandAndMatch"
    if configODS.get('runAllPromptsOnLLM'):
        global llm
        if not llm: llm = LLM()
        for dir_path in os.listdir(configODS.get('promptsPath')):
            for file_path in os.listdir(configODS.get('promptsPath') + dir_path):
                if file_path.endswith('.json'):
                    promptsPath = configODS.get('promptsPath') + dir_path + '/' + file_path
                    llmMatchedFilePath = configODS.get('llmOutcomePath') + dir_path + '/' + file_path
                    promptDict = utilsODS.importFromJson(promptsPath)
                    promptResult = {}
                    for promptKey in tqdm(promptDict, desc=f'running prompts for {file_path.replace(".json", "")}'):
                        prompt = promptDict.get(promptKey)
                        yesOrNo = llm.get_prediction(prompt)
                        promptResult[promptKey] = yesOrNo
                    utilsODS.saveToJson(promptResult, llmMatchedFilePath)

    """
    generates final matching:
                                1. exact matches are used as initial newMatches
                                2. collect neighbors for each pair in newMatches
                                3. get LLM outcome for cross product of all collected neighbors (using generated prompts in promptsPath)
                                4. match 'yes' pairs to a (n:m)-mapping
                                5. compute (1:1)-mapping for matches
                                6. update newMatches to contain only newMatches of this iteration
                                7. jump to 2. if newMatches != empty

    """
    if configODS.get('runMissingPromptsOnDemandAndMatch'):
        for dir_path in os.listdir(configODS.get('promptsPath')):
            if os.path.isdir(configODS.get('promptsPath') + '/' + dir_path):
                for file_path in tqdm(os.listdir(configODS.get('promptsPath') + dir_path), desc=f'matching {dir_path}'):
                    if file_path.endswith('.json'):
                        llmMatchedFilePath = configODS.get('llmOutcomePath') + dir_path + '/' + file_path
                        bipartiteMatchingPath = configODS.get('bipartiteMatchingPath') + dir_path + '/' + file_path
                        
                        #prepare bipartite graph
                        verticesL = set()
                        verticesR = set()
                        edges = {}
                        alreadyMatched = {}#contains all nodes which have a final match
                        exactMatches = utilsODS.importFromJson(configODS.get('exactMatchPath') + file_path)
                        
                        #pre match the exact matches
                        for key1, key2, _ in exactMatches:
                            alreadyMatched[key1] = key2     #prevent key1 from getting matched to anything else in ontology2
                            alreadyMatched[key2] = key1     #prevent key2 from getting matched from anything else in ontology1
                            edges[key1] = [key2]
                            verticesL.add(key1)
                            verticesR.add(key2)
                        
                        #depending on the threshold for exact matches it could be a (n:m)-mapping => compute (1:1)-mapping
                        newMatches = generateMaximumBipartiteMatching.findMaximumBipartiteMatching(list(verticesL), list(verticesR), edges)
                        finalMatches = newMatches

                        #prepare variables and load values
                        ontoName1, node1 = key1.split('#')
                        ontoName2, node2 = key2.split('#')
                        onto1 = ontologies.get(ontoName1)
                        onto2 = ontologies.get(ontoName2)
                        neighborhoodRange = configODS.get('neighborhoodRange')
                        alignmentFilePath = configODS.get('similarityPath') + file_path
                        triples = utilsODS.importFromJson(alignmentFilePath)

                        #start actual neighborhood matching process
                        i = 0
                        while len(newMatches) > 0:
                            i += 1
                            newNeighborhoodMatches = []
                            for currentNeighborhoodRange in range(1, 1 + neighborhoodRange):#start by matching near neighborhood and extend range
                                #prepare bipartite graph
                                verticesL = set()
                                verticesR = set()
                                edges = {}
                                possibleAlignments = []

                                #collect matching candidates using cross product of not matches neighbors
                                for key1, key2 in newMatches:
                                    _, node1 = key1.split('#')
                                    _, node2 = key2.split('#')
                                    neighborsOf1 = [keyA for keyA in getNeighbors(node1, onto1, currentNeighborhoodRange) if not alreadyMatched.get(keyA)]
                                    neighborsOf2 = [keyB for keyB in getNeighbors(node2, onto2, currentNeighborhoodRange) if not alreadyMatched.get(keyB)]
                                    possibleAlignments += [(keyA, keyB) for keyA in neighborsOf1 for keyB in neighborsOf2 if triples.get(keyA + ';' + keyB) and triples.get(keyA + ';' + keyB)[2] >= configODS.get('thresholdForConsideration')]
                                
                                #get LLM results for match or no match and build (n:m)-mapping
                                for keyA, keyB in tqdm(possibleAlignments, desc = f'running prompts for {currentNeighborhoodRange}-hop neighborhoods in {ontoName1}-{ontoName2}'):
                                    if not alreadyMatched.get(keyA) and not alreadyMatched.get(keyB):
                                        if getLLMPrediction(keyA, keyB, llmMatchedFilePath) == 'yes':
                                            verticesL.add(keyA)
                                            verticesR.add(keyB)
                                            if not edges.get(keyA):
                                                edges[keyA] = []
                                            edges[keyA].append(keyB)
                                
                                #compute (1:1)-mapping and add them to final matching
                                newNeighborhoodMatches = generateMaximumBipartiteMatching.findMaximumBipartiteMatching(list(verticesL), list(verticesR), edges)
                                finalMatches += newNeighborhoodMatches
                                
                                #update alreadyMatched variable
                                for keyX, keyY in newNeighborhoodMatches:
                                    alreadyMatched[keyX] = keyY
                                    alreadyMatched[keyY] = keyX
                            newMatches = newNeighborhoodMatches
                        utilsODS.saveToJson(finalMatches, bipartiteMatchingPath)

    #export found matches to a .rdf file
    if configODS.get('exportFinalMatchingsToRDF'):
        for dir_path in os.listdir(configODS.get('bipartiteMatchingPath')):
            for file_path in os.listdir(configODS.get('bipartiteMatchingPath') + dir_path):
                if file_path.endswith('.json'):
                    bipartiteMatchingPath = configODS.get('bipartiteMatchingPath') + dir_path + '/' + file_path
                    rdfPath = configODS.get('rdfPath') + dir_path + '/' + file_path.replace('.json', '') + '.rdf'
                    bipartiteMatchedClasses = utilsODS.importFromJson(bipartiteMatchingPath)
                    t = [('http://' + key1, 'http://' + key2, '=', 1.0) for key1, key2 in bipartiteMatchedClasses]
                    utilsODS.saveToJson('placeholder for rdf content', rdfPath, doPrint=False)
                    serialize_mapping_to_file(rdfPath, t)
                    print('exported ' + rdfPath)
main()