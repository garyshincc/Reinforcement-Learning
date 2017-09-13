import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import keras
import collections
import os
import sys

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json

import numpy as np
from scipy import optimize

class Neural_Network(object):
	def __init__(self, Lambda = 0):
		# we have num hours of workout and num calories
		# a 4 layer network
		self.inputLayerSize = 4
		self.outputLayerSize = 1
		self.firstLayerSize = 3
		self.secondLayerSize = 3

		self.W1 = np.random.randn(self.inputLayerSize,self.firstLayerSize)
		self.W2 = np.random.randn(self.firstLayerSize,self.secondLayerSize)
		self.W3 = np.random.randn(self.secondLayerSize,self.outputLayerSize)

		self.Lambda = Lambda

		self.trainX = None
		self.trainY = None
		self.trainYarr = []
		self.memory = collections.deque(maxlen=2000)

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001

	def forward(self, x):
		self.z2 = np.dot(x, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		self.a3 = self.sigmoid(self.z3)
		self.z4 = np.dot(self.a3, self.W3)
		yHat = self.sigmoid(self.z4)
		print yHat
		return yHat

	def costFunction(self, x, y):
		self.yHat = self.forward(x)
		J = 0.5*sum((y-self.yHat)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
		return J


	def costFunctionPrime(self, x, y):
		self.yHat = self.forward(x)

		delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
		dJdW3 = np.dot(self.a3.T, delta4) + self.Lambda*self.W3

		delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
		dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2

		delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(x.T, delta2) + self.Lambda*self.W1
		return dJdW1, dJdW2, dJdW3

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		# derivative of sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)

	def getParams(self):
        # get W1, W2 and W3 Rolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
		return params
    
	def setParams(self, params):
		#Set W1 and W2 using single parameter vector:
		W1_start = 0
		W1_end = self.firstLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], \
			(self.inputLayerSize, self.firstLayerSize))

		W2_end = W1_end + self.firstLayerSize*self.secondLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], \
			(self.firstLayerSize, self.secondLayerSize))

		W3_end = W2_end + self.secondLayerSize*self.outputLayerSize
		self.W3 = np.reshape(params[W2_end:W3_end], \
			(self.secondLayerSize, self.outputLayerSize))

	def computeGradients(self, x, y):
		dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(x,y)
		return np.concatenate((dJdW1.ravel(),dJdW2.ravel(),dJdW3.ravel()))

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

		target = reward
		prediction = self.forward(next_state)

		self.trainX = np.array(state)
		self.trainY = np.array([target])

class Trainer(object):
	def __init__(self, N):
		# local reference to neural network
		self.N = N

	def costFunctionWrapper(self, params, x, y):
		self.N.setParams(params)
		cost = self.N.costFunction(x,y)
		grad = self.N.computeGradients(x,y)
		return cost, grad

	def callBackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.N.trainX, self.N.trainY))

	def train(self):
		self.J = []

		params0 = self.N.getParams()
		options = {'maxiter':500, 'disp': True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, \
			jac = True, method='BFGS', args=(self.N.trainX,self.N.trainY), \
			options = options, callback=self.callBackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res


class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.num_episodes = 200
		self.nn = Neural_Network()
		self.trainer = Trainer(self.nn)

	def visualize(self, observation):
		obs_dict = dict.fromkeys([0,1,2,3])
		for i, sample in enumerate(observation):
			obs_dict[i] = sample
		plt.clf()
		plt.ylim((-1, 1))
		plt.bar(range(len(obs_dict)), obs_dict.values())
		plt.xticks(range(len(obs_dict)), obs_dict.keys())
		plt.plot()
		plt.pause(0.01)

	def load_model(self):
		pass

	def run_episode(self, num_episodes=200, num_frames=1000):
		print ("Running " + str(num_episodes) + " episodes, " + str(num_frames) + " frames each")

		for episode in range(num_episodes):
			state = self.env.reset()
			state = np.reshape(state, [1, 4])

			for frame in range(num_frames):
				self.env.render()
				action = int(self.nn.forward(state)[0][0] + 0.5)

				observation, reward, done, info = self.env.step(action)
				next_state = np.reshape(observation, [1, 4])

				self.nn.remember(state, action, reward, next_state, done)

				state = next_state

				if (done):
					print ("episode: {}/{}, score: {}".format(episode, num_episodes, frame))
					break

			self.trainer.train()

	def play(self):
		while (True):
			state = self.env.reset()
			state = np.reshape(state, [1, 4])

			while (True):
				self.env.render()
				action = self.agent.act(state)

				observation, reward, done, info = self.env.step(action)
				next_state = np.reshape(observation, [1, 4])
				state = next_state

				if (done):
					print ("Episode completed")
					break


def main():
	cart_pole = CartPole()
	if (not cart_pole.load_model()):
		cart_pole.run_episode()
	cart_pole.play()


if __name__ == '__main__':
	main()

