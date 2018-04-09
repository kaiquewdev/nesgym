import tensorflow as tf

from model import q_function

class QFunctionImageInputTest(tf.test.TestCase):
	def instanceComparison(self):
		with self.test_session():
			self.assertEqual(q_function((84,84,1),16).__class__,tf.keras.layers.Input)

class QFunctionActionsTest(tf.test.TestCase):
	def instanceComparison(self):
		with self.test_session():
			self.assertEqual(q_function((84,84,1),16).__class__,tf.keras.layers.Dense)

class QModelTest(tf.test.TestCase):
	def instanceComparison(self):
		with self.test_session():
			self.assertEqual(q_model((84,84,1),16).__class__,tf.keras.models.Model)

if __name__ == '__main__':
	tf.test.main()