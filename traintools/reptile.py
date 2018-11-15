#reptile example on keras
# Original code on pytorch at https://blog.openai.com/reptile/
# Porting to keras by shi3z Apr. 12 2018
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from keras import backend as K
from copy import deepcopy
from model import multitask_cnn, loss_dict, loss_weights_dict


seed = 0
plot = True
innerstepsize = 0.02 # stepsize in inner SGD
innerepochs = 1 # number of epochs of each inner SGD
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = 30000 # number of outer updates; each iteration we sample one task and update on it
rng = np.random.RandomState(seed)
base_lr = 0.001


def perform_reptile_initialization(model):


	#TODO: REPLACE ALL THESE TASKS WITH OMNIGLOT IMAGES FOR PRETRAINING
	# Define task distribution
	x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points
	ntrain = 10 # Size of training minibatches

	#compile our conv net
	optimizer = RMSprop(lr=base_lr)	
	model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights_dict, metrics=["accuracy"])

	f_plot = gen_task()
	xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]
	

	# Reptile training loop
	for iteration in range(niterations):
    		weights_before = model.weights

    		# Generate task
    		f = gen_task()
    		y_all = f(x_all)
    		
		# Do SGD on this task
    
    		inds = rng.permutation(len(x_all))
    		for _ in range(innerepochs):
        		for start in range(0, len(x_all), ntrain):
            			mbinds = inds[start:start+ntrain]
            			train_on_batch(model, x_all[mbinds], y_all[mbinds])


    		weights_after = model.weights

    		outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    		for i in range(len(weights_after)):
			model.weights[i]  = (weights_before[i]+ (weights_after[i]-weights_before[i])*outerstepsize)

    		if plot and iteration==0 or (iteration+1) % 10 == 0:
        		plt.cla()
        		f = f_plot
   		     	weights_before = model.weights
        		plt.plot(x_all, model.predict(x_all), label="pred after 0", color=(0,0,1))
        		for inneriter in range(32):
            			train_on_batch(xtrain_plot, f(xtrain_plot))
            			if (inneriter+1) % 8 == 0:
                			frac = (inneriter+1) / 32
                			plt.plot(x_all, model.predict(x_all), label="pred after %i"%(inneriter+1), color=(frac, 0, 1-frac))
        		plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
                	lossval = np.square(model.predict(x_all) - f(x_all)).mean()
        		plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        		plt.ylim(-4,4)
        		plt.legend(loc="lower right")
        		plt.pause(0.01)
        		for i in range(len(model.weights)):
            			model.weights[i] = weights_before[i]
        		print("-----------------------------")
        		print("iteration: ", (iteration+1))
        		print("loss on plotted curve: {lossval:.3f}") 

        return model

def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine




def train_on_batch(model,x,y):
    model.train_on_batch(x,y)
   


