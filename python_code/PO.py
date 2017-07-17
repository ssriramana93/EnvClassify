import libexperiment
import lstmProg
d = lstmProg.l

def PO(nPolIter, nExpIter, experiment):
	for pIter in xrange(nPolIter):
		for eIter in xrange(nExpIter):	
			ret = experiment.execute()