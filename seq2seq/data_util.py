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
import json
import collections

url = 'http://mattmahoney.net/dc/'

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data

'''
cluster_by_len = {}
if not os.path.exists('cluster_data.json'):
    text = read_data(filename).strip(' ').split(' ')
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    for word in text:
        if len(word) not in cluster_by_len:
            cluster_by_len[len(word)]=set()
        cluster_by_len[len(word)].add(word)
    for ite in cluster_by_len:
        cluster_by_len[ite]=list(cluster_by_len[ite])
    cluster_by_len=collections.OrderedDict(sorted(cluster_by_len.items()))
    with open('cluster_data.json', 'w') as outfile:
        json.dump(cluster_by_len, outfile)
else:
    with open('cluster_data.json') as data_file:
        cluster_by_len = json.load(data_file)

times=0
for ite in cluster_by_len:
    print (ite)
    #print(len(cluster_by_len[ite]))
    times+=1
    if times == 5:
        break
'''

#ave_len=sum([len(ite) for ite in text])/len(text)
#print (ave_len)

'''
vocabulary_size = len(string.ascii_lowercase)+1  # [a-z] + end tag， use end tag to represent the end
first_letter = ord(string.ascii_lowercase[0])
'''


def char2id(char,first_letter):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter
    else:
        return -1


def id2char(dictid,first_letter):
    if dictid >= 0:
        return chr(dictid + first_letter)
    else:
        return '  '
    #else:
     #   return 'EOW'
