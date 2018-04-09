import tensorflow as tf

from model import q_function
# from model import q_model
# from model import DoubleDQN

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
			self.assertEqual(q[1],tf.Tensor)

# class QModelTest(tf.test.TestCase):
# 	def instanceComparison(self):
# 		with self.test_session():
# 			self.assertEqual(q_model((84,84,1),16).__class__,tf.keras.models.Model)

# class DoubleDQNClassTest(tf.test.TestCase):
# 	def setUp(self):
# 		self.dqn = DQN(image_shape=(84, 84, 1),
#                        num_actions=16,
#                        training_starts=10000,
#                        target_update_freq=4000,
#                        training_batch_size=64,
#                        exploration=exploration_schedule)

if __name__ == '__main__':
	tf.test.main()