import tflearn
import tensorflow as tf
import numpy as np
class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, maxseq_length, n_agents, i_agent, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.seq_len = maxseq_length
        self.n_agents = n_agents
        self.i_agent = i_agent
        self.learning_rate = learning_rate
        self.tau = tau
        self.curr_seq = tf.placeholder(shape = [None, None], dtype = tf.int32)
        # Create the critic network
        self.inputs, self.action, self.actions, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_actions, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)
        #self.action_grads2 = tf.gradients(self.out, self.actions)

        print self.action_grads



    def create_critic_network(self):
        self.sa_dim = self.s_dim + self.a_dim
        inputs = tflearn.input_data(shape=[None, self.seq_len, self.sa_dim])
        #actions = tflearn.input_data(shape=[None, self.a_dim*self.n_agents])
        actions = [tflearn.input_data(shape=[None, self.a_dim]) for i in xrange(self.n_agents)]

        #action = tf.slice(actions, [0, self.i_agent*self.a_dim], [-1,self.a_dim])
        actions_vec = tf.concat(actions, axis = 1)
        #net = tflearn.fully_connected(inputs, 400, activation='relu')

        net = tflearn.reshape(inputs, new_shape=[-1, self.sa_dim])
        ip_nunits = 300
        net = tflearn.fully_connected(net, ip_nunits, weights_init=tflearn.initializations.xavier())
        net = tflearn.reshape(net, new_shape=(-1, self.seq_len, ip_nunits))

        net = tflearn.gru(net, 400, activation='relu',return_seq = True, dynamic = True, weights_init = tflearn.initializations.xavier())
        net = tf.concat(net, axis = 0)

        op_nunits = 300
        net = tflearn.fully_connected(net, op_nunits, weights_init = tflearn.initializations.xavier())
        net = tflearn.reshape (net, new_shape = (-1, self.seq_len, op_nunits))

        net = tf.gather_nd(net, self.curr_seq)

        net = tflearn.reshape(net, new_shape = (-1, op_nunits))

        aip_dim = 300
        t2 = tflearn.fully_connected(actions_vec, aip_dim)
        net = tflearn.merge([net, t2], mode = 'concat')
       # net = tflearn.activation(
       #     tf.matmul(net, net.W) + tf.matmul(actions, t2.W) + t2.b, activation='relu')

        out = tflearn.fully_connected(net,1)









        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        #t1 = tflearn.fully_connected(net, 300)
        #t2 = tflearn.fully_connected(action, 300)

        #net = tflearn.activation(
        #    tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        #w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, actions[self.i_agent], actions, out

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

    def train(self, inputs, actions, predicted_q_value, seq_idx):
        a_dict = {i: d for i, d in zip(self.actions,actions)}
        feed_dict = {
            self.inputs: inputs,
            self.curr_seq: seq_idx,
            self.predicted_q_value: predicted_q_value
        }
        feed_dict.update(a_dict)
        return self.sess.run([self.out, self.optimize], feed_dict = feed_dict)

    def predict(self, inputs, actions, seq_idx):
        a_dict = {i: d for i, d in zip(self.actions, actions)}
        feed_dict = {
            self.inputs: inputs,
            self.curr_seq: seq_idx
        }
        feed_dict.update(a_dict)
        return self.sess.run(self.out, feed_dict=feed_dict)

    def predict_target(self, inputs, actions, seq_idx):
        a_dict = {i: d for i, d in zip(self.target_actions, actions)}
        print inputs.shape
        feed_dict = {
            self.target_inputs: inputs,
            self.curr_seq: seq_idx
        }
        feed_dict.update(a_dict)
        return self.sess.run(self.target_out, feed_dict= feed_dict)

    def action_gradients(self, inputs, actions, seq_idx):
        a_dict = {i: d for i, d in zip(self.actions, actions)}
        feed_dict = {
            self.inputs: inputs,
            self.curr_seq: seq_idx
        }
        feed_dict.update(a_dict)
        return self.sess.run(self.action_grads, feed_dict=feed_dict)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
