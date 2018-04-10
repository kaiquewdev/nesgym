import tensorflow as tf

from keras.models import Model

from model import q_function
from model import q_model
from model import DoubleDQN
from model import ReplayBuffer

import numpy as np

class QFunctionImageInputTest(tf.test.TestCase):
	def testInstanceComparison(self):
		with self.test_session():
			q = q_function((84,84,1),16)
			self.assertEqual(type(q[0]),tf.Tensor)

class QFunctionActionsTest(tf.test.TestCase):
	def testInstanceComparison(self):
		with self.test_session():
			q = q_function((84,84,1),16)
			self.assertEqual(type(q[1]),tf.Tensor)

class QModelTest(tf.test.TestCase):
	def testInstanceComparison(self):
		with self.test_session():
			q = q_model((84,84,1),16)
			self.assertEqual(type(q),Model)

class DoubleDQNClassTest(tf.test.TestCase):
	def setUp(self):
		self.dqn = DoubleDQN(image_shape=(84, 84, 1),
                       num_actions=16,
                       training_starts=10000,
                       target_update_freq=4000,
                       training_batch_size=64,
                       exploration=100)

	def testInstanceComparison(self):
		self.assertEqual(type(self.dqn),DoubleDQN)

	def testInsidePropertyReplayBufferInstanceComparison(self):
		self.assertEqual(type(self.dqn.replay_buffer),ReplayBuffer)

	def testChooseActionMethodOutputType(self):
		observations = np.array([10,10,10])
		self.assertEqual(type(self.dqn.choose_action(5,observations)),int)

	def testGetLearningRateMethod(self):
		self.assertEqual(self.dqn.get_learning_rate() > 0,True)

if __name__ == '__main__':
	tf.test.main()