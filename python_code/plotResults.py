import cPickle
import numpy as np
from matplotlib import pyplot as plt
Results = cPickle.load(open("Results.p","rb"))
rewards = {}
for traj in Results:
	for ep in traj:
		ao, ac ,r = ep
		for key in r:
			if key not in rewards:
				rewards[key] = [np.sum(r[key])]
			else:
				rewards[key].append(np.sum(r[key]))

plt_data = []
for key in rewards:
	#plt_data.append(rewards[key])
	plt.scatter(np.arange(len(rewards[key])), rewards[key])
#plt.plot(plt_data)
plt.show()