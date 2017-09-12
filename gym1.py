import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import keras
import collections
import os
import sys

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = collections.deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def load_model(self):
		if os.path.isfile("deepQNet.json"):
			print ("I've found a model in your directory.")
			response = raw_input("Should I attempt to load it? (y/n) ")
			try:
				json_file = open("deepQNet.json", "r")
				json_model = json_file.read()
				json_file.close()
				self.model = model_from_json(json_model)
				print ("Loaded model")
				self.model.load_weights("deepQNet.h5")
				print ("Loaded weights")
				return True
			except:
				print ("Error: " + str(sys.exc_info()[0]))
				return False
	
	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action


	def replay(self, batch_size):
		try:
			minibatch = random.sample(self.memory, batch_size)
		except ValueError:
			minibatch = random.sample(self.memory, len(self.memory))
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save_model(self):
		print ("Training complete.")
		model_json = self.model.to_json()
		with open("deepQNet.json", "w") as open_file:
			open_file.write(model_json)
		print ("Model saved!")
		self.model.save_weights("deepQNet.h5")
		print ("Weights saved!")



class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.agent = DQNAgent(4, 2)

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
		self.agent.load_model()

	def run_episode(self, num_episodes=250, num_frames=1000):
		print ("Running " + str(num_episodes) + " episodes, " + str(num_frames) + " frames each")

		for episode in range(num_episodes):
			state = self.env.reset()
			state = np.reshape(state, [1, 4])

			for frame in range(num_frames):
				self.env.render()
				action = self.agent.act(state)

				observation, reward, done, info = self.env.step(action)
				next_state = np.reshape(observation, [1, 4])

				self.agent.remember(state, action, reward, next_state, done)

				state = next_state

				if (done):
					print ("episode: {}/{}, score: {}".format(episode, num_episodes, frame))
					print ("Episode completed")
					break

			self.agent.replay(128)

	def save_model(self):
		self.agent.save_model()

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
		cart_pole.save_model()
	cart_pole.play()


if __name__ == '__main__':
	main()











