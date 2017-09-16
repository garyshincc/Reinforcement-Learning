import gym
import os
import sys
import random
import time
import collections

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import load_model
from keras.models import model_from_json

import numpy as np



class cartPoleModelAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.discount_rate = 0.95
		self.learning_rate = 0.01
		self.learning_decay = 0.01
		self.memory = collections.deque(maxlen=20000)

	def load_model(self):
		if os.path.isfile("cartPoleModel.json"):
			print ("I've found a model in your directory.")
			response = input("Should I attempt to load it? (y/n):")
			if (response == 'y'):
				try:
					json_file = open("cartPoleModel.json", "r")
					json_model = json_file.read()
					json_file.close()
					self.model = model_from_json(json_model)
					print ("Loaded model")
					self.model.load_weights("cartPoleModel.h5")
					print ("Loaded weights")
					time.sleep(1)
					return True
				except:
					print ("Error: " + str(sys.exc_info()[0]))

		return False

	def save_model(self):
		print ("Training complete.")
		model_json = self.model.to_json()
		with open("cartPoleModel.json", "w") as open_file:
			open_file.write(model_json)
		print ("Model saved!")
		self.model.save_weights("cartPoleModel.h5")
		print ("Weights saved!")

		print ("Saving full train version.")
		trainX_batch = []
		trainY_batch = []
		for sample in self.memory:
			state, action, reward, done, next_state = sample
			target = reward
			if (not done):
				target = reward + self.discount_rate * np.max(self.model.predict(np.reshape(next_state, [1,4]))[0])
			train_target = self.model.predict(np.reshape(state, [1,4]))
			train_target[0][action] = target
			trainX_batch.append(np.reshape(state, [1,4])[0])
			trainY_batch.append(train_target[0])
			
		self.model.fit(np.array(trainX_batch), np.array(trainY_batch), batch_size=len(trainX_batch), epochs=800)

		model_json = self.model.to_json()
		with open("cartPoleModelFull.json", "w") as open_file:
			open_file.write(model_json)
		print ("Model saved!")
		self.model.save_weights("cartPoleModelFull.h5")
		print ("Weights saved!")

	def create_model(self):
		self.model = Sequential()
		self.model.add(Dense(units=24, input_shape=(4,), activation='tanh'))
		self.model.add(Dense(units=48, activation='tanh'))
		self.model.add(Dense(units=self.action_size, activation='linear'))
		self.model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.learning_decay), loss='mse')

	def remember(self, observation):
		self.memory.append(observation)

	def reinforce(self, review_episodes):
		try:
			batch = random.sample(self.memory, review_episodes)
		except ValueError:
			batch = random.sample(self.memory, len(self.memory))

		#print ("training a batch of: {} memory cells".format(len(batch)))
		trainX_batch = []
		trainY_batch = []
		for sample in batch:
			state, action, reward, done, next_state = sample
			target = reward

			# goal is to increase the 'previous state' with
			# best find of next state
			if (not done):
				target = reward + self.discount_rate * np.max(self.model.predict(np.reshape(next_state, [1,4]))[0])
			train_target = self.model.predict(np.reshape(state, [1,4]))
			train_target[0][action] = target
			#print ("train_target after: {}".format(train_target))
			trainX_batch.append(np.reshape(state, [1,4])[0])
			trainY_batch.append(train_target[0])
		self.model.fit(np.array(trainX_batch), np.array(trainY_batch), batch_size=len(trainX_batch), epochs=1, verbose=0)

	def action(self, state):
		action = self.model.predict(np.reshape(state, [1,4]))
		#print ("action: {}".format(action))
		action = np.argmax(action[0])
		return action



class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.agent = cartPoleModelAgent(self.env.observation_space.shape[0], self.env.action_space.n)
		self.num_reinforce = 128
		self.exploration_rate = 1
		self.exploration_decay = 0.995
		self.min_exploration_rate = 0.01

	def load_model(self):
		model_loaded = self.agent.load_model()
		if (not model_loaded):
			self.agent.create_model()
		return model_loaded

	def save_model(self):
		self.agent.save_model()

	def run_episode(self, num_episodes=300, num_frames=300):
		print ("Running {} episodes, {} frames each".format(num_episodes,num_frames))

		for episode in range(num_episodes):
			state = self.env.reset()

			for frame in range(num_frames):
				#self.env.render()
				action = self.agent.action(state)
				if (np.random.rand() < self.exploration_rate):
					action = self.env.action_space.sample()

				observation, reward, done, info = self.env.step(action)

				status = (state, action, reward, done, observation)
				self.agent.remember(status)

				state = observation
				if (done):
					print ("episode: {}/{}, score: {}".format(episode, num_episodes, frame))
					break

			if (self.exploration_rate > self.min_exploration_rate):
				self.exploration_rate *= self.exploration_decay
			self.agent.reinforce(self.num_reinforce)

	def play(self):
		while (True):
			state = self.env.reset()

			while (True):
				self.env.render()
				action = self.agent.action(state)
				self.env.step(action)
				if (done):
					print ("Episode completed")
					break


def main():
	cart_pole = CartPole()
	model_loaded = cart_pole.load_model()
	if (not model_loaded):
		print ("Could not find model. Creating new model.")
		cart_pole.run_episode()
		cart_pole.save_model()
	cart_pole.play()


if __name__ == '__main__':
	main()
