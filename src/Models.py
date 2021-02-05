import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Softmax

class MindReader(tf.keras.Model):
	def __init__(self, out_dim, **kwargs):
		super(MindReader, self).__init__(**kwargs)
		self.d1 = Dense(units=out_dim)
		self.sm = Softmax()

	def call(self, inputs: tf.Tensor, softmax: bool = True) -> tf.Tensor:
		"""
		NN forward pass
		Args:
			inputs: assume shape [mb_size, features]

		Returns: shape [mb_size, out_dim]
		"""
		z = self.d1(inputs)
		return self.sm(z) if softmax else z
