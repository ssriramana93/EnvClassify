import tflearn
import tensorflow as tf
import numpy as np
from tflearn.data_utils import pad_sequences
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, maxseq_length, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.max_seq = maxseq_length
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.nenv = 3
        self.curr_seq = tf.placeholder(shape = [None, None], dtype = tf.int32)


        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])



        # Combine the gradients here
        self.actor_gradients = tf.gradients(
         tf.squeeze(self.scaled_out), self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
    
        self.sa_dim = self.s_dim + self.a_dim
        inputs = tflearn.input_data(shape=[None, self.max_seq, self.sa_dim]) ## (samples, timesteps, ip_dim)
        #inputs = tflearn.input_data(shape=[None, None, self.sa_dim]) ## (samples, timesteps, ip_dim)
        #seq_len = tf.shape(inputs)[1]

        net = tflearn.reshape (inputs, new_shape = [-1, self.sa_dim])
        net = tflearn.fully_connected(net, 300, weights_init = tflearn.initializations.xavier())
        net = tflearn.reshape (net, new_shape = (-1, self.max_seq, 300))
        #print "net1", net
      #  net = tflearn.layers.recurrent.gru(inputs, self.a_dim, activation='relu',return_seq = True)
        net = tflearn.gru(net, 400, activation='relu',return_seq = True, dynamic = False, weights_init = tflearn.initializations.xavier())

        net = tf.concat(net, axis = 0)

        #print "net2", net

        #net = tflearn.reshape (net, new_shape = (-1, 400))
        net = tflearn.fully_connected(net, self.a_dim, weights_init = tflearn.initializations.xavier())

        ##Apply softmax only to the last self.nenv entries
        net = tf.concat([tf.slice(net, [0, 0], [-1, self.a_dim - self.nenv]), tflearn.layers.core.activation(tf.slice(net, [0, self.a_dim - self.nenv], [-1, self.nenv] ), activation = 'softmax')], axis = 1)
        net = tflearn.reshape (net, new_shape = (-1, self.max_seq, self.a_dim))

        #out = tf.slice(net, self.curr_seq, [-1, 1, -1]
        out = tf.gather_nd(net, self.curr_seq)
        scaled_out = out
        print scaled_out
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        #w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #out = tflearn.fully_connected(
        #    net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        #scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient, seq_idx):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.curr_seq: seq_idx
        })
    '''
    def pad_with_zeros(self, inputs):
        ##Pad this shit
        if (inputs.ndim == 2):
            inputs = np.expand_dims(inputs, axis = 0)
        #inputs = pad_sequences(inputs, maxlen=self.max_seq, dtype='float32', padding='post', truncating='post',
        #                                 value=np.zeros((1,inputs.shape[2])))
        nsamp = inputs.shape[0]
        seq_len = inputs.shape[1]
        print seq_len
        dseq_len = self.max_seq - seq_len
        assert(dseq_len >= 0)
        if not dseq_len:
            return inputs, seq_len

        #inputs = inputs.tolist()
       # print inputs
       # inputs = [ip + [0.0]*dseq_len for ip in inputs]
       # print inputs

        inputs = np.concatenate([inputs, np.zeros((nsamp, self.max_seq - seq_len, self.sa_dim))], axis = 1)
        return inputs, seq_len
    '''


    def predict(self, inputs, seq_idx):
        #inputs, seq_len = self.pad_with_zeros(inputs)
        #print  seq_len
        out = self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs, self.curr_seq: seq_idx
        })
        print "out", out.shape
        #return out[:,seq_len - 1,:]
        return out

    def predict_target(self, inputs, seq_idx):

        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs, self.curr_seq: seq_idx
        })


    def predict_sequence(self, inputs):
        inputs = self.pad_with_zeros(inputs)
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })[:, -1, :]

    def predict_target_sequence(self, inputs):
        inputs = self.pad_with_zeros(inputs)
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })



    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
