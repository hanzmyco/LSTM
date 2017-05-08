# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
from data_util import *

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings,vocabulary_size,first_letter):
        self._text = text
        self._text_size = len(text)
        self.vocabulary_size=vocabulary_size
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        self.next_position=0
        self.begin_index=0
        self.first_letter=first_letter

    def _next_batch(self):
        # generate one batch
        batch_encoder = np.zeros(shape=(self._batch_size, self.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            if self._text_size > self.begin_index+b:  # meet enough word
                if len(self._text[self.begin_index+b]) >self.next_position:
                    batch_encoder[b, char2id(self._text[self.begin_index+b][self.next_position],self.first_letter)] = 1.0
                #else: #
                 #   batch_encoder[b,0]=1.0  # padding with ' ', 看看这个词够不够长

        return batch_encoder

    def next(self):
        #generate number of unrolling
        batches_encoder = []
        self.next_position=0
        for step in range(self._num_unrollings):
            batches_encoder.append(self._next_batch())
            self.next_position+=1
        #batches_encoder.append(np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float))
        #for b in range(self._batch_size):  # last node in encoder
        #    batches_encoder[self._num_unrollings][b,0]=1.0
        batches_decoder=[]
        # mirror image of batch encoder,  self becomes fles
        for step in xrange(self._num_unrollings):
            batches_decoder.append(np.zeros(shape=(self._batch_size, self.vocabulary_size), dtype=np.float))
            for b in range(self._batch_size):
                for i in xrange(0,self.vocabulary_size):
                    batches_decoder[step][b][i]=batches_encoder[self._num_unrollings-step-1][b][i]
        self.begin_index += self._batch_size

        return batches_encoder, batches_decoder


def test_batch_generator(batch_generator,vocabulary_size,first_letter):
    batch_generator.next()
    data=batch_generator.next()
    phrase = ''
    p2=''
    for i in xrange(0,batch_generator._num_unrollings):
            for j in xrange(0,vocabulary_size):
                if data[0][i][0][j]==1.0:
                    phrase+=id2char(j,first_letter)
                    break
            for j in xrange(0,vocabulary_size):
                if data[1][i][0][j]==1.0:
                    p2+=id2char(j,first_letter)
    print (batch_generator._text[64])
    print (phrase)
    print (p2)

def test_visualization1(batch,vocabulary_size,first_letter):
    output_sents=[]
    for b in xrange(0, len(batch[0])):
        phrase_in=''
        phrase_out=''
        for i in xrange(0, num_unrollings):
            #phrase_in+=characters(input_mats[i][b])
            for j in xrange(0,vocabulary_size):
                if batch[0][i][b][j]==1.0:
                    phrase_in+=id2char(j,first_letter)
            for j in xrange(0,vocabulary_size):
                if batch[1][i][b][j]==1.0:
                    phrase_out+=id2char(j,first_letter)

        if phrase_in[::-1] !=phrase_out:
            output_sents.append((phrase_in[::-1],phrase_out))
    print (output_sents)

def test_visualization(concate_input,concate_predict,vocabulary_size,first_letter,num_unrollings):

    input_mats=np.split(concate_input,num_unrollings,0)
    predict_mats=np.split(concate_predict,num_unrollings,0)
    output_sents=[]
    for b in xrange(0, len(input_mats[0])):
        phrase_in=''
        phrase_out=''
        for i in xrange(0, num_unrollings):
            #phrase_in+=characters(input_mats[i][b])
            for j in xrange(0,vocabulary_size):
                if input_mats[i][b][j]==1.0:
                    phrase_in+=id2char(j,first_letter)
            max_index=0
            for j in xrange(0,vocabulary_size):
                if predict_mats[i][b][j]>=predict_mats[i][b][max_index]:
                    max_index=j
            if predict_mats[i][b][max_index]!=0:
                phrase_out+=id2char(max_index,first_letter)
            #phrase_out+=characters(predict_mats[i][b])

        output_sents.append((phrase_in[::-1],phrase_out))
    print (output_sents)
