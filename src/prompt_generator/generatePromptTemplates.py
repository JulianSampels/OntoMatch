import sys
sys.path.append('..')
from src import utils

"""
imports alignments from json_file_path

Parameters:
json_file_path (str): the path to the alignment file

Returns: 
alignments
"""
def importAlignments(json_file_path):
    data = utils.importFromJson(json_file_path)
    if data == None:
        raise Exception(f"problems importing triples from '{json_file_path}'")
    return data


"""
imports context from json_file_path

Parameters:
json_file_path (str): the path to the context file

Returns: 
context
"""
def importContext(json_file_path = './triples_randomWalk_verbalized_out/RandomWalk_verbalized_out.json'):
    data = utils.importFromJson(json_file_path)
    if data == None:
        raise Exception(f"problems importing context from '{json_file_path}'")
    return data

#provides context in a dict
def formatContext(data):
    context = {}
    for i in data:
        key = list(i.keys())[0]
        context.update({key : i.get(key)})
    return context

"""
calls the generate prompt function for required prompt version

Parameters:
alignmentPath (str): the path to the alignment file
contextPath (str): the path to the context file
promptVersion (int): version of prompt to be generated
promptCounter (int): 
                    < 0     => generate all prompts as list
                    >= 0    => generate prompt with index promptCounter
skipIfNoContext (bool): 
                    True    => skip alignment if their is not enough context
                    False   => generate anyways

Returns: 
prompt (str) or list of prompts ([str])
"""
def getPrompt(alignmentPath, contextPath, promptVersion = 0, promptCounter = -1, skipIfNoContext = True):
    triples = importAlignments(alignmentPath)
    context = formatContext(importContext(contextPath))
    #choose correct function for promptVersion
    if promptVersion == 0:
        generatePrompt = lambda triples, context, promptCounter : prompt0(triples, context, promptCounter)
    elif promptVersion == 1:
        generatePrompt = lambda triples, context, promptCounter : prompt1(triples, context, promptCounter)
    elif promptVersion == 2:
        generatePrompt = lambda triples, context, promptCounter : prompt2(triples, context, promptCounter)
    elif promptVersion == 3:
        generatePrompt = lambda triples, context, promptCounter : prompt3(triples, context, promptCounter)
    
    #skipIfNoContext
    if (skipIfNoContext):
        tmpTriples = []
        for i, (key1, key2, _) in enumerate(triples):
            cont1 = context.get(key1)
            cont2 = context.get(key2)
            if (cont1 != None and cont2 != None):
                tmpTriples.append(triples[i])
        triples = tmpTriples

    #generate one or all prompts
    if (promptCounter < 0 or promptCounter >= len(triples)):
        prompt = []
        for promptCounter in range(len(triples)):
            p = generatePrompt(triples, context, promptCounter)
            if (len(p) > 0):
                prompt.append(p)
    else:
        prompt = generatePrompt(triples, context, promptCounter)
        
    return prompt

#function for prompt version 0
def prompt0(triples, context, promptCounter):
    key1, key2, cont1, cont2, onto1, onto2, node1, node2 = extract(triples, context, promptCounter)
    problemDefinition = 'In this task, we are given two concepts along with their definitions from two ontologies.'
    objective = 'Our objective is to provide ontology mapping for the provided ontologies based on their semantic similarities.'
    prompt = problemDefinition + '\n' + objective + '\n'    
    prompt += f'{key1}: {cont1}\n' if (cont1 != None) else ''
    prompt += f'{key2}: {cont2}\n' if (cont2 != None) else ''
    prompt += f'Does the concept "{node1}" correspond to the concept "{node2}"? yes or no:'
    #workaround
    if (onto1 == onto2):
        return ''
    return prompt

#function for prompt version 1
def prompt1(triples, context, promptCounter):
    key1, key2, cont1, cont2, onto1, onto2, node1, node2 = extract(triples, context, promptCounter)
    prompt = 'Classify if two concepts refer to the same real word entity.\n'
    prompt += f'This is the context for the first concept "{node1}":\n'
    prompt += f'{cont1}\n\n' if (cont1 != None) else ''
    prompt += f'This is the context for the second concept "{node2}":\n'
    prompt += f'{cont2}\n' if (cont2 != None) else ''
    prompt += f'Do these concepts "{node1}" and "{node2}" refer to the same real word entity? yes or no:'
    #workaround
    if (onto1 == onto2):
        return ''
    return prompt

#function for prompt version 2
def prompt2(triples, context, promptCounter):
    key1, key2, cont1, cont2, onto1, onto2, node1, node2 = extract(triples, context, promptCounter)
    prompt = 'Classify if the following two concepts are the same.\n'
    prompt += f'###First concept {node1}:\n'
    prompt += f'{cont1}\n' if (cont1 != None) else ''
    prompt += f'###Second concept {node2}:\n'
    prompt += f'{cont2}\n' if (cont2 != None) else ''
    prompt += 'Answer yes or no:'
    #workaround
    if (onto1 == onto2):
        return ''
    return prompt

#function for prompt version 3
def prompt3(triples, context, promptCounter):
    key1, key2, cont1, cont2, onto1, onto2, node1, node2 = extract(triples, context, promptCounter)
    prompt = f'Is {node1} and {node2} the same?\n'
    prompt += f'The answer which can be yes or no is:'
    #workaround
    if (onto1 == onto2):
        return ''
    return prompt

#extract necessary information for generating prompt
def extract(triples, context, promptCounter):
    key1 = triples[promptCounter][0]
    key2 = triples[promptCounter][1]
    onto1, node1 = key1.split("#")
    onto2, node2 = key2.split("#")
    cont1 = context.get(key1)
    cont2 = context.get(key2)
    return key1, key2, cont1, cont2, onto1, onto2, node1, node2

#saves prompts in a list to a json file
def savePromptToJson(promptList, json_path):
    utils.saveToJson(promptList, json_path)


#p = getPrompt("../../results/result_alignments/conference/alignments.json", "../../results/result_triples/triples_randomTree_verbalized_out.json")
#p = getPrompt(4)
#print('\n\n'.join(p) if (type(p) == type([])) else p)
