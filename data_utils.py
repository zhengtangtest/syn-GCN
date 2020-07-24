import json
from collections import defaultdict
import numpy as np
np.random.seed(1)
import math
from model import *

EOS_token = 0

id2relation = ['no_relation', 'org:top_members/employees', 'per:religion', 'org:founded', 'per:cities_of_residence', 'per:stateorprovince_of_birth', 'per:alternate_names', 'org:dissolved', 'per:children', 'org:stateorprovince_of_headquarters', 'per:date_of_birth', 'org:parents', 'org:member_of', 'org:shareholders', 'org:founded_by', 'org:subsidiaries', 'per:siblings', 'per:cause_of_death', 'per:city_of_birth', 'per:date_of_death', 'org:city_of_headquarters', 'org:alternate_names', 'org:members', 'per:country_of_birth', 'org:website', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:political/religious_affiliation', 'per:other_family', 'per:parents', 'per:schools_attended', 'per:age', 'per:origin', 'org:country_of_headquarters', 'per:country_of_death', 'per:stateorprovinces_of_residence', 'per:employee_of', 'org:number_of_employees/members', 'per:spouse', 'per:charges', 'per:countries_of_residence', 'per:title']
relation2id = {r:i for i, r in enumerate(id2relation)}

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"EOS":0,"UNK":1}
        self.index2word = {0: "EOS", 1:"UNK"}
        self.n_words = 2 # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def load_embeddings(file, lang):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            x = np.array([float(i) for i in line_split[1:]])
            vector = (x /np.linalg.norm(x))
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
            break
    base = math.sqrt(6/embedding_size)
    emb_matrix = np.random.uniform(-base,base,(lang.n_words, embedding_size))
    # for i in range(3, lang.n_words):
    #     word = lang.index2word[i]
    #     if word in emb_dict:
    #         emb_matrix[i] = emb_dict[word]
    return emb_matrix, embedding_size

def makeIndexes(lang, seq):
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    return indexes

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def load_data(file_name, input_lang=None, dep_lang=None):

    is_train = False

    s = set()

    if input_lang is None:
        input_lang  = Lang("input")
        dep_lang    = Lang("dep")
        is_train    = True

    data = list()

    for datapoint in json.load(open(file_name)):
        
        relation    = datapoint['relation']
        s.add(relation)
        sentence    = ['$ROOT$'] + datapoint['token']
        
        edge_index  = [list(range(1, len(datapoint['stanford_head'])+1)), 
                    datapoint['stanford_head']]
        edge_label  = datapoint['stanford_deprel']

        entity_subj = list(range(datapoint['subj_start']+1, datapoint['subj_end']+2))
        entity_obj  = list(range(datapoint['obj_start']+1, datapoint['obj_end']+2))
        type_subj   = datapoint['subj_type']
        type_obj    = datapoint['obj_type']

        if is_train:
            input_lang.addSentence(sentence)
            dep_lang.addSentence(edge_label)
        sent_index  = makeIndexes(input_lang, sentence)
        label_index = makeIndexes(dep_lang, edge_label)
        
        data.append((relation2id[relation], sent_index, edge_index, label_index, entity_subj, entity_obj,
            type_subj, type_obj))

    if is_train:
        return input_lang, dep_lang, data
    else:
        return data