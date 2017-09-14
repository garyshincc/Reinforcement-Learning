import gym
import numpy as np
import random
import tensorflow as tf




class FrozenLake:
	def __init__(self):
		self.env = gym.make('FrozenLake-v0')
		self.learning_rate = 0.1
		self.decay_params = 0.99
		self.num_episodes = 1000
		self.r_list = list()
		self.exploration_rate = 0.7
		self.exploration_decay = 0.99
		self._init_model()
		self.one_hot_identity_matrix = np.identity(16)

	def _init_model(self):
		tf.reset_default_graph()
		# input one-hot vector of state
		self.inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
		self.weights = tf.Variable(tf.random_uniform([16,4]), dtype=tf.float32)
		self.out_q = tf.matmul(self.inputs1, self.weights)
		# get max value!
		self.predict = tf.argmax(self.out_q, 1)

		self.next_q = tf.placeholder(shape=[1,4], dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.next_q - self.out_q))
		trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.update_model = trainer.minimize(loss)

	def one_hot(self, state):
		return self.one_hot_identity_matrix[state: state+1]

	def start(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)

			for i in range(self.num_episodes):
				prev_state = self.env.reset()
				net_reward = 0
				done = False

				for j in range(100):
					if (np.random.rand(1) < self.exploration_rate):
						action = self.env.action_space.sample()
					else:
						action, all_q = sess.run([self.predict, self.out_q], feed_dict={
							self.inputs1: self.one_hot(prev_state)
						})
						action = action[0]
					# act!
					state, reward, done, info = self.env.step(action)
					# get derivative values
					next_q = sess.run([self.out_q], feed_dict={
						self.inputs1: self.one_hot(state)
						})

					# maximum likely q value is....!!!
					max_q = np.max(next_q)
					target_q = sess.run([self.out_q], feed_dict={
							self.inputs1: self.one_hot(prev_state)
						})[0]
					target_q[0, action] = reward + self.decay_params * max_q

					# train!
					update_model, weights = sess.run([self.update_model, self.weights], feed_dict={
						self.inputs1: self.one_hot(state),
						self.next_q: target_q
						})


					net_reward += reward
					prev_state = state
					if done:
						print ("Episode {}/{} complete.".format(i, self.num_episodes))
						self.exploration_rate *= self.exploration_decay
						break
				self.r_list.append(net_reward)

			print ("Percent of success: " + str(sum(self.r_list)/self.num_episodes) + "%")



def main():
	frozenlake = FrozenLake()
	frozenlake.start()

if __name__ == "__main__":
	main()