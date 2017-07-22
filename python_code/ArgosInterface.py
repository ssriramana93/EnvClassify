import numpy as np
import tensorflow as tf
import actor
import tflearn
from copy import deepcopy
class ArgosInterface:
	def __init__(self):
#	def __init__(self):
		self.obs = {}
		self.ac = {}
		self.reward = {}
 		self.oa_seq = {}
		self.oar_seq = {}
		self.oa_hists = {}
		self.id = None
		self.traj = []
		self.results = []

	def RegisterActorAndNoise(self, actor, noise_fn, replay_buffer):
		self.actor = actor
		self.noise_fn = noise_fn
		self.replay_buffer = replay_buffer

	

	def Add2Dict(self, dictionary, id, value):
		if (id in dictionary) and (len(dictionary[id]) >= (self.actor.max_seq - 1)):
			traj = (deepcopy(self.oa_hists), deepcopy(self.ac), deepcopy(self.reward))
			self.results.append(traj)
			self.replay_buffer.add(traj)
			self.ClearAllDict()

		if id not in dictionary:
			dictionary[id] = []
		dictionary[id].append(value)	

	def GetfromDict(self, dictionary, id):
		if id not in dictionary:
			return []
		else:
			return deepcopy(dictionary[id])


	def SetInput(self, temp, id):
		self.id = id
		assert(len(temp) == self.actor.s_dim)

		#self.InitDict(self.obs, id)
		#self.obs[id].append(np.array(temp))
		self.Add2Dict(self.obs, id, np.array(temp))
		#self.obseq.append(self.obs)
		

	def GetPrevAction(self):
		#if self.id not in self.ac:
		#	return np.zeros(self.actor.a_dim)
		#else:
		#	return self.ac[self.id][-1]
		return np.zeros(self.actor.a_dim)

	#def SetOutput(self, temp):
	#	self.ac = temp	

	def ComputeOutput(self):
		prev_ac = self.GetPrevAction()
		oa = np.concatenate((self.obs[self.id][-1], prev_ac))
		oa_seq = self.GetfromDict(self.oa_seq, self.id)
		oa_seq.append(oa)

		oa_seq = np.array(oa_seq).reshape((1, -1, self.actor.sa_dim))
		seq_idx = np.array([[0, oa_seq.shape[1] - 1]])


		oa_seq = np.concatenate([oa_seq, np.zeros((1, self.actor.max_seq - oa_seq.shape[1], self.actor.sa_dim))], axis = 1)

		action = self.noise_fn(self.actor.predict(oa_seq, seq_idx))[0] ## Input full sequence of observation history as list of (o0, a0, o1, a1, a2, ...)  h = (a0o1, a1o2, a2o3, )
	    #self.InitDict(self.ac, self.id)
	    #self.ac[self.id].append(action)
		#print action
		self.Add2Dict(self.oa_hists, self.id, oa_seq)
		self.Add2Dict(self.ac, self.id, action)

	def GetOutput(self):

		ac = self.ac[self.id][-1].tolist()
		#print "action", ac
		assert(len(ac) == self.actor.a_dim)
		return ac

	def GetTraj(self):
		results = deepcopy(self.results)
		self.results = []
		return results

	def SetReward(self, reward, id):
		assert(self.id == id)
	   # self.InitDict(self.reward, self.id)
	   #	self.reward[id].append(reward)
		self.Add2Dict(self.reward, id, reward)
	    #self.InitDict(self.oar_seq, self.id)
		#self.oar_seq[id].append((np.concatenate((self.obs[id][-1], self.ac[id][-1]), axis = 1),reward))
		#self.history[id].append([self.obs, self.ac, self.reward])
		self.Add2Dict(self.oa_seq, id, np.concatenate((self.obs[id][-1], self.ac[id][-1])))
		#print ("id", len(self.oa_seq[self.id]), self.oa_seq[self.id])

		#self.Add2Dict(self.oar_seq, id, (np.concatenate((self.obs[id][-1], self.ac[id][-1])),reward))
		#self.Add2Dict(self.oar_seq, id, (self.obs[id][-1], self.ac[id][-1], reward))



	def ClearAllDict(self):
		self.ac.clear()
		self.obs.clear()
		self.oa_seq.clear()
		self.oa_hists.clear()
		self.oar_seq.clear()
		self.reward.clear()