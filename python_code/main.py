import sys
sys.path.append("/home/argos3-examples/python_code")


# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 5000
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
MIN_HISTORY_LEN = 80

# ===========================
#   Actor and Critic DNNs
# ===========================



import tensorflow as tf
import numpy as np
import tflearn
import time
from actor import *
from critic import *
import libexperiment
from replay_buffer import *
import ArgosInterfaceObject
import cPickle



def noise_fn(action):

	return action

env = ArgosInterfaceObject.argos_interface
myexp = libexperiment.experiment()

state_dim = 67
action_dim = 14
action_bound = 2
maxseq_len = 301
n_robots = 1
n_envs = 2

sess = tf.Session()

replay_buffer = ReplayBuffer(BUFFER_SIZE, MIN_HISTORY_LEN)
actor = ActorNetwork(sess, state_dim, action_dim, n_envs, action_bound, maxseq_len,
                             ACTOR_LEARNING_RATE, TAU)

critic_vec = []
for i in xrange(n_robots):
	critic_vec.append(CriticNetwork(sess, state_dim, action_dim, maxseq_len, n_robots, i,
						   CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars()))

env.RegisterActorAndNoise(actor, noise_fn, replay_buffer)
actor = env.actor

sess.__enter__()  # equivalent to `with sess:`
tf.global_variables_initializer().run()

Traj = []
nIter = 2
pickle_interval = 10
actor.reset_target_network()
for i in xrange(n_robots):
	critic_vec[i].reset_target_network()

for ep in xrange(MAX_EPISODES):
	print "___________iteration =", ep, "_____________"

	start = time.time()
	myexp.execute()
	end = time.time()
	print ("Time to GetTraj:", end - start)
	traj = env.GetTraj()

	Traj.append(traj)
	if not (ep % 10):
		start = time.time()
		cPickle.dump(Traj, open("Results.p", "wb"))
		end = time.time()
		print ("Time to Pickle this shit:", end - start)

	env.ClearAllDict()


	for n in xrange(nIter):
		for i  in xrange(n_robots):
			critic = critic_vec[i]
			oa_batch, a_batch, r_batch, oa2_batch, seq = env.replay_buffer.sample_batch(MINIBATCH_SIZE)
			assert(len(oa_batch) == len(oa2_batch))
			action_next_vec = []
			seq_idx = None
			start = time.time()

			for j in xrange(n_robots):

				id = np.arange(len(oa2_batch[j])).tolist()
				seq_idx = np.array(zip(id, seq))
				#print seq_idx
				action_next = actor.predict_target(np.squeeze(oa2_batch[j]), seq_idx)
				action_next_vec.append(action_next)
			end = time.time()
			print ("Time to Predict Actor:", end - start)
			#action_vec = np.concatenate(action_vec, axis = 1)
			#print action_vec.shape,  r_batch
			#print [action_vec[0].shape]
			oa_curr = np.squeeze(oa_batch[i])
			oa_next = np.squeeze(oa2_batch[i])
			start = time.time()
			q_next, ips = critic.predict_target(oa_next, action_next_vec, seq_idx)
			end = time.time()
			print ("Time to Predict Critic:", end - start)

			print ips[0]
			#print q_next.shape
			predicted_q = np.reshape(r_batch[i], q_next.shape) + GAMMA*q_next

			action_vec = [np.reshape(a_batch[j], action_next_vec[0].shape) for j in xrange(n_robots)]

			start = time.time()
			_,_,loss = critic.train(oa_curr, action_vec, predicted_q, seq_idx)
			end = time.time()
			print ("Time to Train Critic:", end - start)
			print ("Critic Loss", loss)



			a_outs = actor.predict(oa_curr, seq_idx)
			action_vec[i] = a_outs


			grads = critic.action_gradients(oa_curr, action_vec, seq_idx)


			start = time.time()
			actor.train(oa_curr, grads[0], seq_idx)
			end = time.time()
			print ("Time to Train Actor:", end - start)


			# Update target networks
			actor.update_target_network()
			critic.update_target_network()



	for i, t in enumerate(traj):
		o, a, r = t
		print "Trial ", np.mean([np.sum(r[key]) for key in r])

#action = actor.predict()

	#myexp.destroy();
