import tensorflow as tf

class _LayersOverride(object):
	def __init__(self, kernel_regularizer, bias_regularizer):
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer

	def Conv2D(self, filters, kernel_size, **kwargs):
		kwargs['kernel_regularizer'] = self.kernel_regularizer
		kwargs['bias_regularizer'] = self.bias_regularizer

		name = kwargs.get('name')
		if 'conv5_block1' in name:
			kwargs['strides'] = 1

		return tf.keras.layers.Conv2D(filters, kernel_size, **kwargs)

	# Forward all non-overridden methods to the keras layers
	def __getattr__(self, item):
		return getattr(tf.keras.layers, item)

def preprocess_input(image):
	return tf.keras.applications.imagenet_utils.preprocess_input(image, mode='tf')

def ResNet50FeatureExtractor(
		kernel_regularizer=None,
		bias_regularizer=None,
		**kwargs):
	"""
	Instantiates the ResNet50 architecture (Modified for object detection).
	
	This wraps the ResNet50 tensorflow Keras application, but uses the
	Keras application's kwargs-based monkey-patching API to override the Keras
	architecture with the following changes:
	- Changes the default output stride to 16 (32 in the original model).
	- Adds support for regularization.
	Args:
		- kernel_regularizer: Initializer for the kernel weights matrices.
		- bias_regularizer: Initializer for the bias vectors.
		- **kwargs: Keyword arguments forwarded directly to the
			`tf.keras.applications.ResNet50` method that constructs 
			the Keras model.
	Returns:
	  A Keras model instance.
	"""

	layers_override = _LayersOverride(
		kernel_regularizer=kernel_regularizer,
		bias_regularizer=bias_regularizer)

	return tf.keras.applications.ResNet50(
		layers=layers_override,
		include_top=False,
		weights='imagenet',
		**kwargs)

