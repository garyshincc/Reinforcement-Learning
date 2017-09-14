import gym
import numpy as np


'''
Q-Learning is designed to provide algorithm that learns long-term expected rewards.

Bellman equation:
expected long-term reward is equal to the reward from the current
action plus expected reward from the best future action
taken at the following state.
'''

class FrozenLake:
	def __init__(self):
		self.env = gym.make('FrozenLake-v0')
		self.learning_rate = 0.8
		self.decay_params = 0.95
		self.num_episodes = 2000
		self.r_list = list()
		self._init_Q_table()

	def _init_Q_table(self):
		self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
		print ("Q Table: {}".format(self.q_table))

	def start(self):
		for i in range(self.num_episodes):
			prev_state = self.env.reset()
			net_reward = 0
			done = False
			j = 0

			while j < 99:
				j+= 1

				action = np.argmax(self.q_table[prev_state,:] + np.random.randn(1, self.env.action_space.n)*(1./(i+1)))
				state, reward, done, info = self.env.step(action)
				
				adjustment = self.learning_rate * (reward + self.decay_params * np.argmax(self.q_table[state,:]) - self.q_table[prev_state,action])

				self.q_table[prev_state, action] = self.q_table[prev_state, action] + adjustment
				net_reward += reward
				prev_state = state
				if done:
					print ("Episode {}/{} complete.".format(i, self.num_episodes))
					break
			self.r_list.append(net_reward)
		print ("Final Q table results:\n" + str(self.q_table))


def main():
	frozenlake = FrozenLake()
	frozenlake.start()

if __name__ == "__main__":
	main()