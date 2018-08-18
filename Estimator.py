from pkgs import *

class Estimator():


	def __init__(self, scope = "estimator", summaries_dir = None):

		self.scope = scope

		''' The FileWriter class provides a mechanism to create an event file in a given directory and add summaries and events to it. 
			The class updates the file contents asynchronously. This allows a training program to call methods to add data 
			to the file directly from the training loop, without slowing down training.
		'''
		self.summary_writer = None

		with tf.variable_scope(scope):

			self.build_model()

			if(summaries_dir):
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))

				if(not os.path.exists(summary_dir)):
					os.makedirs(summary_dir)

				self.summary_writer = tf.summary.FileWriter(summary_dir)


	def build_model(self):

		''' Our input is a 4 frames of shape 84, 48'''
		self.X = tf.placeholder(shape = [None, 84, 84, 4], dtype = tf.uint8, name = "X")

		''' The TD target value '''
		self.Y = tf.placeholder(shape = [None], dtype = tf.float32, name = "Y")
			
		''' Integer id of which action was selected '''
		self.actions = tf.placeholder(shape = [None], dtype = tf.int32, name = "actions")

		''' Preprocessing '''
		X = tf.to_float(self.X) / 255.0
		batch_size = tf.shape(self.X)[0]

		''' 3 Convolutional layers '''
		conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn = tf.nn.relu)
		conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn = tf.nn.relu)
		conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn = tf.nn.relu)

		''' Fully connected layers '''
		flattened = tf.contrib.layers.flatten(conv3)
		fc1 = tf.contrib.layers.fully_connected(flattened, 512)
		self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

		''' Get predictions of the chosen actions only. [-1] to flatten the tensor '''
		gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions 
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

		''' Calculate loss '''
		self.losses = tf.squared_difference(self.Y, self.action_predictions)
		self.loss = tf.reduce_mean(self.losses)

		''' Optimization '''
		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op = self.optimizer.minimize(self.loss, global_step = tf.contrib.framework.get_global_step())

		''' Summaries for Tensorboard '''
		self.summaries = tf.summary.merge([
											tf.summary.scalar("loss", self.loss),
											tf.summary.histogram("loss_hist", self.losses),
											tf.summary.histogram("q_values_list", self.predictions),
											tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))])


	def predict(self, sess, s):

		'''
		Args:
			sess: Tensorflow session.
			s: state input of shape [batch_size, 4, 160, 160, 3]

		Returns:
			Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the
			estimated action values.
		'''

		return sess.run(self.predictions, {self.X: s})

	def update(self, sess, s, a, y):

		'''
		Args:
			sess: Tensorflow session object
			s: state input of shape [batch_size, 4, 160, 160, 3]
			a: chosen actions of shape [batch_size]
			y: Targets of shape [batch_size]

		Returns:
			The calculated loss on the batch.
		'''

		feed_dict = {self.X: s, self.Y: y, self.actions: a}

		summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
													feed_dict)

		if(self.summary_writer):
			self.summary_writer.add_summary(summaries, global_step)

		return loss