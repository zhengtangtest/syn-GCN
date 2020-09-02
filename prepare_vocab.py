"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('data_dir', help='TACRED directory.')
    parser.add_argument('vocab_dir', help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)

    train_rules = load_rules(train_file)
    dev_rules = load_rules(dev_file)
    test_rules = load_rules(test_file)

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab, args.min_freq)
    r = build_rule_vocab(train_rules)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    with open(vocab_file, 'wb') as outfile:
        pickle.dump((v, r), outfile)
    np.save(emb_file, embedding)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            tokens += d[2]
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def load_rules(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            tokens += d[6]
    print("{} rule tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + ['SUBJ', 'OBJ'] + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_rule_vocab(tokens):
    counter = Counter(t for t in tokens)
    v = sorted([t for t in counter], key=counter.get, reverse=True)
    print("rule vocab built with {}/{} tokens.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched


if __name__ == '__main__':
    main()


