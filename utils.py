#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math

import numpy as np


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words.
    @param pad_token (int): padding token.
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        NOTE: The correct return type should be (list[list[int]]). It's using the word's index in the embedding,
        not the actual words. 
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    # Find the longest sentence's length. 
    longest_len = 0
    for l in sents:
      if len(l) >= longest_len:
        longest_len = len(l)

    for l in sents:
      padded_sent = l + [pad_token] * (longest_len - len(l))
      sents_padded.append(padded_sent)
    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def test_pad_sents():
  sents = [[1,2,3], [2,3,4,5,6,7],[7,8]]
  pad_token = 0
  sents_padded = pad_sents(sents, pad_token)
  assert sents_padded == [[1, 2, 3, 0, 0, 0], [2, 3, 4, 5, 6, 7], [7, 8, 0, 0, 0, 0]], "pad sentence failed"
  print("pad sentence passed")

if __name__ == '__main__':
    test_pad_sents()
