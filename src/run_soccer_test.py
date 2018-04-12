import tensorflow as tf

from gym.wrappers import Monitor

from run_soccer import get_env

class RunSoccerGetEnvMethodTest(tf.test.TestCase):
	def testGetEnvType(self):
		with self.test_session():
			self.assertEqual(type(get_env()),Monitor)

if __name__ == '__main__':
	tf.test.main()