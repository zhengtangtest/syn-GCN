"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
ROOT_TOKEN = '<ROOT>'
ROOT_ID = 2

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN, ROOT_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14, '<ROOT>': 15}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46, '<ROOT>':47}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

DEPREL_bi_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct_in': 2, 'compound_in': 3, 'case_in': 4, 'nmod_in': 5, 'det_in': 6, 'nsubj_in': 7, 'amod_in': 8, 'conj_in': 9, 'dobj_in': 10, 'ROOT_in': 11, 'cc_in': 12, 'nmod:poss_in': 13, 'mark_in': 14, 'advmod_in': 15, 'appos_in': 16, 'nummod_in': 17, 'dep_in': 18, 'ccomp_in': 19, 'aux_in': 20, 'advcl_in': 21, 'acl:relcl_in': 22, 'xcomp_in': 23, 'cop_in': 24, 'acl_in': 25, 'auxpass_in': 26, 'nsubjpass_in': 27, 'nmod:tmod_in': 28, 'neg_in': 29, 'compound:prt_in': 30, 'mwe_in': 31, 'parataxis_in': 32, 'root_in': 33, 'nmod:npmod_in': 34, 'expl_in': 35, 'csubj_in': 36, 'cc:preconj_in': 37, 'iobj_in': 38, 'det:predet_in': 39, 'discourse_in': 40, 'csubjpass_in': 41, 'punct_out': 42, 'compound_out': 43, 'case_out': 44, 'nmod_out': 45, 'det_out': 46, 'nsubj_out': 47, 'amod_out': 48, 'conj_out': 49, 'dobj_out': 50, 'ROOT_out': 51, 'cc_out': 52, 'nmod:poss_out': 53, 'mark_out': 54, 'advmod_out': 55, 'appos_out': 56, 'nummod_out': 57, 'dep_out': 58, 'ccomp_out': 59, 'aux_out': 60, 'advcl_out': 61, 'acl:relcl_out': 62, 'xcomp_out': 63, 'cop_out': 64, 'acl_out': 65, 'auxpass_out': 66, 'nsubjpass_out': 67, 'nmod:tmod_out': 68, 'neg_out': 69, 'compound:prt_out': 70, 'mwe_out': 71, 'parataxis_out': 72, 'root_out': 73, 'nmod:npmod_out': 74, 'expl_out': 75, 'csubj_out': 76, 'cc:preconj_out': 77, 'iobj_out': 78, 'det:predet_out': 79, 'discourse_out': 80, 'csubjpass_out': 81}

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

INFINITY_NUMBER = 1e12

KEYS = ['words', 'mask','e_mask', 'pos', 'ner', 'deprel', 'd_mask', 'subj_mask', 'obj_mask', 'edge_index']
