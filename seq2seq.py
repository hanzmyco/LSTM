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

url = 'http://mattmahoney.net/dc/'

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        self.next_position=0
        self.begin_index=0

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch_encoder = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            if self._text_size > self.begin_index+b:  # meet enough word
                if len(self._text[self.begin_index+b]) >self.next_position:
                    batch_encoder[b, char2id(self._text[self.begin_index+b][self.next_position])] = 1.0
                else: #
                    batch_encoder[b,0]=1.0  # padding with ' ', 看看这个词够不够长

        return batch_encoder

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches_encoder = []
        self.next_position=0
        for step in range(self._num_unrollings):
            batches_encoder.append(self._next_batch())
            self.next_position+=1
        batches_encoder.append(np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float))
        for b in range(self._batch_size):
            batches_encoder[self._num_unrollings][b,1]=1.0
        batches_decoder=[]
        # mirror image of batch encoder,  self becomes fles
        for step in xrange(self._num_unrollings):
            batches_decoder.append(np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float))
            for b in range(self._batch_size):
                for i in xrange(0,vocabulary_size):
                    batches_decoder[step][b][i]=batches_encoder[self._num_unrollings-step-1][b][i]
        self.begin_index += batch_size

        return batches_encoder, batches_decoder

def test_batch_generator(batch_generator):
    batch_generator.next()
    data=batch_generator.next()
    phrase = ''
    p2=''
    for i in xrange(0,batch_generator._num_unrollings):
            for j in xrange(0,vocabulary_size):
                if data[0][i][0][j]==1.0:
                    phrase+=id2char(j)
                    break
            for j in xrange(0,vocabulary_size):
                if data[1][i][0][j]==1.0:
                    p2+=id2char(j)
    print (batch_generator._text[64])
    print (phrase)
    print (p2)


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

text = read_data(filename).strip(' ').split(' ')

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

#ave_len=sum([len(ite) for ite in text])/len(text)
#print (ave_len)

vocabulary_size = len(string.ascii_lowercase)+2  # [a-z] + ' ' + end tag， use end tag to represent the end
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 2
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 1


def id2char(dictid):
    if dictid > 1:
        return chr(dictid + first_letter - 2)
    elif dictid==0:
        return ' '
    else:
        return 'EOW'


#print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
#print(id2char(1), id2char(26), id2char(0))

def logprob(predictions,labels):
    predictions[predictions<1e-10]=1e-10
    return np.sum(np.multiply(labels,-np.log(predictions)))/labels.shape[0]

def sample_distribution(distribution):
    r = random.uniform(0,1)
    s = 0
    for i in xrange(len(distribution)):
        s+=distribution[i]
        if s >=r:
            return i
    return len(distribution)-1

def sample(prediction):
    p= np.zeros(shape=[1,vocabulary_size],dtype=np.float)
    p[0,sample_distribution(prediction[0])]=1.0
    return p

def random_distribution():
    b=np.random.uniform(0.0,1.0,size=[1,vocabulary_size])
    return b/np.sum(b,1)[:,None]

num_nodes=64
graph = tf.Graph()
batch_size = 64
num_unrollings=5
batch_g=BatchGenerator(text,batch_size,num_unrollings)
test_batch_generator(batch_g)


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)


with graph.as_default():

    # Input gate: input, previous output, and bias.
    ix_encoder=tf.Variable(tf.truncated_normal([vocabulary_size,num_nodes],-0.1,0.1))
    im_encoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib_encoder = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx_encoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm_encoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb_encoder = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx_encoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm_encoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb_encoder = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox_encoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om_encoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob_encoder = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output_encoder = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state_encoder = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)


    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))

    #gate_x= tf.Variable(tf.truncated_normal([vocabulary_size,num_nodes*4],-0.1,0.1))
    #gate_m=tf.Variable(tf.truncated_normal([num_nodes,4*num_nodes],-0.1,0.1))
    #gate_b=tf.Variable(tf.zeros([1,4*num_nodes]))

    sx_encoder = tf.concat([ix_encoder, fx_encoder, cx_encoder, ox_encoder],1)
    sm_encoder = tf.concat([im_encoder, fm_encoder, cm_encoder, om_encoder],1)
    sb_encoder = tf.concat([ib_encoder, fb_encoder, cb_encoder, ob_encoder],1)

    def lstm_cell1(i,o,state):
        '''
        gatex=tf.matmul(i,gate_x)
        gatem=tf.matmul(o,gate_m)
        input_gate=tf.sigmoid(gatex[:,:num_nodes]+gatem[:,:num_nodes]+gate_b[0,:num_nodes])
        forget_gate=tf.sigmoid(gatex[:,num_nodes:2*num_nodes]+gatem[:,num_nodes:2*num_nodes]+gate_b[0,num_nodes:2*num_nodes])
        update=gatex[:,2*num_nodes:3*num_nodes]+gatem[:,2*num_nodes:3*num_nodes]+gate_b[0,2*num_nodes:3*num_nodes]
        state=forget_gate*state+input_gate*tf.tanh(update)
        output_gate=gatex[:,3*num_nodes:4*num_nodes]+gatem[:,3*num_nodes:4*num_nodes]+gate_b[0,3*num_nodes:4*num_nodes]
        return output_gate*tf.tanh(state),state
        '''
        smatmul = tf.matmul(i, sx) + tf.matmul(o, sm) + sb
        smatmul_input, smatmul_forget, update, smatmul_output = tf.split(smatmul,4,1)
        input_gate = tf.sigmoid(smatmul_input)
        forget_gate = tf.sigmoid(smatmul_forget)
        output_gate = tf.sigmoid(smatmul_output)
        state = forget_gate * state + input_gate * tf.tanh(update) # final memory cell
        return output_gate * tf.tanh(state), state   # hidden state,  final memory cell


    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state

        # Input data.


    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell1(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.concat(train_labels, 0), logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell1(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))
