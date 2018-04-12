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
		with self.test_session():
			self.assertEqual(type(self.dqn),DoubleDQN)

	def testInsidePropertyReplayBufferInstanceComparison(self):
		with self.test_session():
			self.assertEqual(type(self.dqn.replay_buffer),ReplayBuffer)

	def testGetReplayBufferIdx(self):
		with self.test_session():
			obs = np.array([10,10,10])
			self.assertEqual(type(self.dqn.get_replay_buffer_idx(obs)),int)

	def testChooseActionMethodOutputType(self):
		with self.test_session():
			observations = np.array([10,10,10])
			self.assertEqual(type(self.dqn.choose_action(5,observations)),int)

	# def testChooseActionMethodOutputValue(self):
		# with self.test_session():
			# observations = np.array([10,10,10])
			# self.assertEqual(self.dqn.choose_action(5,observations) > 0,True)

	def testIterationsEvaluationType(self):
		with self.test_session():
			self.assertEqual(type(self.dqn.eval_iters()),np.int64)

	def testMulDecayCrossingEvaluatedIterationsType(self):
		with self.test_session():
			self.assertEqual(type(self.dqn.mul_decay_iters()),np.float32)

	def testNormalizeMultiplicatedParametersValues(self):
		with self.test_session():
			self.assertEqual(type(self.dqn.normalize_params()),np.float64)

	def testGetLearningRateMethod(self):
		with self.test_session():
			self.assertEqual(self.dqn.get_learning_rate() > 0,True)

	def testGetterLeaningRateType(self):
		with self.test_session():
			self.assertEqual(type(self.dqn.get_learning_rate()),np.float32)

	def testFirstStateOfGetAvgLoss(self):
		with self.test_session():
			self.dqn.latest_losses.append(.92781)
			expected_avg_loss = np.mean(np.array(self.dqn.latest_losses,dtype=np.float32))
			self.assertEqual(self.dqn.get_avg_loss(),expected_avg_loss)

	def testGetAvgLossWhenZero(self):
		with self.test_session():
			self.assertEqual(self.dqn.get_avg_loss(),None)

if __name__ == '__main__':
	tf.test.main()