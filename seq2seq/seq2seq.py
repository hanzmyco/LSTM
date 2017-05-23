# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
#import zipfile
#from six.moves import range
#from six.moves.urllib.request import urlretrieve
from batch_generator import *
from data_util import *
from virsualization import * 

url = 'http://mattmahoney.net/dc/'


text = read_data(filename).strip(' ').split(' ')

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

#ave_len=sum([len(ite) for ite in text])/len(text)
#print (ave_len)

vocabulary_size = len(string.ascii_lowercase) # [a-z] + ' ' + end tag， use end tag to represent the end
first_letter = ord(string.ascii_lowercase[0])

#print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
#print(id2char(1), id2char(26), id2char(0))

'''
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
'''
num_nodes=100
graph = tf.Graph()
batch_size = 64
num_unrollings=3
batch_g=BatchGenerator(text,batch_size,num_unrollings,vocabulary_size,first_letter)
test_batch_generator(batch_g,vocabulary_size,first_letter)


train_batches = BatchGenerator(train_text, batch_size, num_unrollings,vocabulary_size,first_letter)
valid_batches = BatchGenerator(valid_text, batch_size, num_unrollings,vocabulary_size,first_letter)


with graph.as_default():
    # encoder layer
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

    # decoder layer
    # Input gate: input, previous output, and bias.
    ix_decoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im_decoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib_decoder = tf.Variable(tf.zeros([1, num_nodes]))
    ic_decoder=tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))


    # Forget gate: input, previous output, and bias.
    fx_decoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm_decoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb_decoder = tf.Variable(tf.zeros([1, num_nodes]))
    fc_decoder=tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))

    # Memory cell: input, state and bias.
    cx_decoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm_decoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb_decoder = tf.Variable(tf.zeros([1, num_nodes]))
    cc_decoder=tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))

    # Output gate: input, previous output, and bias.
    ox_decoder = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om_decoder = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob_decoder = tf.Variable(tf.zeros([1, num_nodes]))
    oc_decoder=tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))




    # Variables saving state across unrollings.
    saved_output_decoder = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state_decoder = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)


    #gate_x= tf.Variable(tf.truncated_normal([vocabulary_size,num_nodes*4],-0.1,0.1))
    #gate_m=tf.Variable(tf.truncated_normal([num_nodes,4*num_nodes],-0.1,0.1))
    #gate_b=tf.Variable(tf.zeros([1,4*num_nodes]))

    sx_encoder = tf.concat([ix_encoder, fx_encoder, cx_encoder, ox_encoder],1)
    sm_encoder = tf.concat([im_encoder, fm_encoder, cm_encoder, om_encoder],1)
    sb_encoder = tf.concat([ib_encoder, fb_encoder, cb_encoder, ob_encoder],1)


    sx_decoder = tf.concat([ix_decoder, fx_decoder, cx_decoder, ox_decoder], 1)
    sm_decoder = tf.concat([im_decoder, fm_decoder, cm_decoder, om_decoder], 1)
    sb_decoder = tf.concat([ib_decoder, fb_decoder, cb_decoder, ob_decoder], 1)
    sc_decoder = tf.concat([ic_decoder,fc_decoder,cc_decoder,oc_decoder],1)

    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))

    y = tf.Variable(tf.zeros([batch_size, vocabulary_size]), trainable=False)

    def lstm_cell_encoder(i,o,state):
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
        smatmul = tf.matmul(i, sx_encoder) + tf.matmul(o, sm_encoder) + sb_encoder
        smatmul_input, smatmul_forget, update, smatmul_output = tf.split(smatmul,4,1)
        input_gate = tf.sigmoid(smatmul_input)
        forget_gate = tf.sigmoid(smatmul_forget)
        output_gate = tf.sigmoid(smatmul_output)
        state = forget_gate * state + input_gate * tf.tanh(update) # final memory cell
        return output_gate * tf.tanh(state), state   # hidden state,  final memory cell

    def lstm_cell_decoder(i,o,state, attention):
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
        smatmul = tf.matmul(i, sx_decoder) + tf.matmul(o, sm_decoder) + tf.matmul(attention,sc_decoder) +sb_decoder
        smatmul_input, smatmul_forget, update, smatmul_output = tf.split(smatmul,4,1)
        input_gate = tf.sigmoid(smatmul_input)
        forget_gate = tf.sigmoid(smatmul_forget)
        output_gate = tf.sigmoid(smatmul_output)
        state = forget_gate * state + input_gate * tf.tanh(update) # final memory cell
        return output_gate * tf.tanh(state), state   # hidden state,  final memory cell



    # Input data.


    train_data_encoder = list()
    train_labels = list()
    for _ in range(num_unrollings):
        train_data_encoder.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    for _ in range(num_unrollings):
        train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))

    #train_inputs = train_data[:num_unrollings]

      # labels are inputs shifted by one time step.

    # Unrolled LSTM loop. encoder
    outputs_encoder = list()
    output_encoder = saved_output_encoder
    state_encoder = saved_state_encoder
    for i in train_data_encoder:
        output_encoder, state_encoder = lstm_cell_encoder(i, output_encoder, state_encoder)
        outputs_encoder.append(output_encoder) # store a list of output
    last_hidden_encoder=output_encoder # last tag of outputs_encoder
    last_cm_encoder=state_encoder


    # Unrolled LSTM loop. decoder
    outputs_decoder = list()
    output_decoder = saved_output_decoder  # pass in the last
    state_decoder = saved_state_decoder
    for i in xrange(0,num_unrollings):
        y = tf.nn.xw_plus_b(tf.concat(output_decoder, 0), w, b)
        output_decoder, state_decoder = lstm_cell_decoder(y, output_decoder, state_decoder,outputs_encoder[num_unrollings-i-1])
        outputs_decoder.append(output_decoder)  # store a list of output



    # State saving across unrollings.
    with tf.control_dependencies([saved_output_encoder.assign(output_encoder),
                                  saved_state_encoder.assign(state_encoder),saved_output_decoder.assign(output_decoder),saved_state_decoder.assign(state_decoder)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs_decoder, 0), w, b)
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


    # Sampling and validation eval: batchsize unrolling.


num_steps = 7001  #how many batches to use
summary_frequency = 100
epoch=10
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for time in xrange(epoch):
      for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings):
          feed_dict[train_data_encoder[i]] = batches[0][i]
          feed_dict[train_labels[i]]=batches[1][i]
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
          labels = np.concatenate(list(batches[1]))
          print('Minibatch perplexity: %.2f' % float(
            np.exp(logprob(predictions, labels))))

          test_visualization(labels,predictions,vocabulary_size,first_letter,num_unrollings)
          #test_visualization1(batches)




      '''
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]

          #reset_sample_state.run()
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
        '''
