from pkgs import *

class StateProcessor():

	'''Processes a raw Atari images. Resizes it and converts it to grayscale'''
	def __init__(self):

		'''A context manager for defining ops that creates variables(layers).
		   This manager validates that the optinal values are from the same graph,
		   ensures that graph is the default graph, and pushes a name scope and
		   a variable scope.
		'''
		with tf.variable_scope("state_processor"):

			''' A placeholder of the raw Atari image dims. '''
			self.input_state = tf.placeholder(shape = [210, 160, 3], dtype = tf.uint8)

			''' Converts one or more images from RGB to grayscale. '''
			self.output = tf.image.rgb_to_grayscale(self.input_state)

			'''	Crops an image to a specified bounding box, it takes
				(img, offset_height, offset_width, target_height, target_width)
				The top-left corner of the returned image is at offset_height, offset_width in image,
				and its lower-right corner is at offset_height + target_height, offset_width + target_width.
			'''
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)

			''' Resizes images to size using the specified method.
				Resized images will be distorted if their original aspect ratio is not the same as size.
			'''
			self.output = tf.image.resize_images(self.output, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

			''' Removes dimensions of size 1 from the shape of a tensor.
				Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed.
			'''
			self.output = tf.squeeze(self.output)


	def process(self, sess, state):

		"""
		Arg:
			sess: S Tensorflow session object.
			state: A [210, 160, 3] Atari RGB state

		Returns:
			A processed [84, 84, 1] state representing greyscale values.
		"""

		return sess.run(self.output, {self.input_state: state}) 