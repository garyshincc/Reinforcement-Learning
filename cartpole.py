import gym
import os
import sys

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense


import numpy as np



class DeepQNetAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.discount_rate = 0.99
		self.learning_rate = 0.001
		self.memory = list()


		self._create_model()

	def load_model(self):
		if os.path.isfile("deepQNet.json"):
			print ("I've found a model in your directory.")
			response = input("Should I attempt to load it? (y/n) ")
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

	def save_model(self):
		print ("Training complete.")
		model_json = self.model.to_json()
		with open("deepQNet.json", "w") as open_file:
			open_file.write(model_json)
		print ("Model saved!")
		self.model.save_weights("deepQNet.h5")
		print ("Weights saved!")

	def _create_model(self):
		self.model = Sequential()
		self.model.add(Dense(units=32, input_shape=(4,), activation='relu'))
		self.model.add(Dense(units=32, activation='relu'))
		self.model.add(Dense(units=32, activation='relu'))
		self.model.add(Dense(units=self.action_size, activation='sigmoid'))
		self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

	def remember(self, observation):
		self.memory.append(observation)

	def reinforce(self, review_episodes):
		

class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.agent = DeepQNetAgent(self.env.observation_space.shape[0], self.env.action_space.n)

	def load_model(self):
		self.agent.load_model()

	def save_model(self):
		self.agent.save_model()

	def run_episode(self, num_episodes=250, num_frames=1000):
		print ("Running " + str(num_episodes) + " episodes, " + str(num_frames) + " frames each")

		for episode in range(num_episodes):
			state = self.env.reset()
			state = np.reshape(state, [1, 4])

			for frame in range(num_frames):
				self.env.render()
				action = self.env.action_space.sample()

				observation, reward, done, info = self.env.step(action)

				if (done):
					print ("episode: {}/{}, score: {}".format(episode, num_episodes, frame))
					print ("Episode completed")
					break

	def play(self):
		while (True):
			state = self.env.reset()

			while (True):
				self.env.render()

				observation, reward, done, info = self.env.step(action)
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
