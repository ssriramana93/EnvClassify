import sys
sys.path.append("/home/argos3-examples/python_code")


# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================



import tensorflow as tf
import numpy as np
import tflearn
from actor import *
from critic import *
import libexperiment
from replay_buffer import *
import ArgosInterfaceObject




def noise_fn(action):
	return action

env = ArgosInterfaceObject.argos_interface
myexp = libexperiment.experiment()

state_dim = 67
action_dim = 15
action_bound = 2
maxseq_len = 101
n_robots = 5
n_envs = 3

sess = tf.Session()

replay_buffer = ReplayBuffer(BUFFER_SIZE)
actor = ActorNetwork(sess, state_dim, action_dim, action_bound, maxseq_len,
                             ACTOR_LEARNING_RATE, TAU)

critic_vec = []
for i in xrange(n_robots):
	critic_vec.append(CriticNetwork(sess, state_dim, action_dim, maxseq_len, n_robots, i,
						   CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars()))

env.RegisterActorAndNoise(actor, noise_fn, replay_buffer)
    
sess.__enter__() # equivalent to `with sess:`
tf.global_variables_initializer().run()



for ep in xrange(MAX_EPISODES):
	myexp.execute()
	traj = env.GetTraj()
	print "traj", len(traj)
	env.ClearAllDict()
	for i  in xrange(n_robots):
		critic = critic_vec[i]
		oa_batch, a_batch, r_batch, oa2_batch, seq = env.replay_buffer.sample_batch(MINIBATCH_SIZE)
		action_next_vec = []
		seq_idx = None
		for j in xrange(n_robots):
			print "oa_batch", len(oa_batch[j])

			id = np.arange(len(oa_batch[j])).tolist()
			seq_idx = np.array(zip(id, seq))
			print seq_idx.shape
			action_next = actor.predict_target(np.squeeze(oa2_batch[j]), seq_idx)
			action_next_vec.append(action_next)
		#action_vec = np.concatenate(action_vec, axis = 1)
		#print action_vec.shape,  r_batch
		#print [action_vec[0].shape]
		oa_full = np.squeeze(oa_batch[i])
		q_next = critic.predict_target(oa_full, action_next_vec, seq_idx)

		print q_next.shape
		predicted_q = np.reshape(r_batch[i], q_next.shape) + GAMMA*q_next

		action_vec = [np.reshape(a_batch[j], action_next_vec[0].shape) for j in xrange(n_robots)]
		print predicted_q
		critic.train(oa_full, action_vec, predicted_q, seq_idx)
		a_outs = actor.predict(oa_full, seq_idx)
		print "a_outs_OK"
		grads = critic.action_gradients(oa_full, action_vec, seq_idx)
		print "grads OK", grads[0].shape
		actor.train(oa_full, grads[0], seq_idx)
		print "actor Trained Really!!??"

		# Update target networks
		actor.update_target_network()
		critic.update_target_network()




#action = actor.predict()

	#myexp.destroy();
