import tensorflow as tf

class InputPipelineCreator(object):
	def __init__(self, num_classes, image_shape, num_steps_per_epoch, max_num_objects):
		'''
		InputPipelineCreator constructor.

		Args:
			- num_classes: Number of classes.
			- image_shape: Shape of the input images. Images that have a different shape
				in the data will be clipped and/or padded to the desired shape.
			- num_steps_per_epoch: Number of images to return in training mode. 
			- max_num_objects: Maximum number of object to keep per image.
		'''
		self.num_classes = num_classes
		self.image_shape = image_shape
		self.num_steps_per_epoch = num_steps_per_epoch
		self.max_num_objects = max_num_objects
		
	def create_input_pipeline(self, filenames, training=True):
		'''
		Create an optimized input pipeline.

		Args:
			- filenames: List of paths to TFRecord data files.
			- training: (Default: True) Boolean value indicating whether the dataset
				will be used for training.

		Returns:
			A tf.data.Dataset object.
		'''
		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.interleave(
			tf.data.TFRecordDataset,
			num_parallel_calls=tf.data.experimental.AUTOTUNE
		)
		dataset = dataset.map(
			self._decode_and_preprocess,
			num_parallel_calls=tf.data.experimental.AUTOTUNE
		)
		dataset = dataset.cache()
		if training:
			dataset = dataset.shuffle(7000)
			dataset = dataset.take(self.num_steps_per_epoch)
			dataset = dataset.map(
				self._augment,
				num_parallel_calls=tf.data.experimental.AUTOTUNE
			)
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
		
		return dataset

	def _augment(self, image, classes, boxes):
		'''
		Randomly perform an horizontal flip with a probability of 0.5.

		Args:
			- image: A single image (shape = [height, width, channels]).
			- classes: The class ids of the objects in the image (one hot encoded).
			- boxes: The bounding boxes coordinates of the objects in the image.

		Returns:
			Three tensors of same type and shape as image, classes, and boxes.
		'''

		def _flip_boxes(boxes):
			ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
			flipped_xmin = tf.subtract(1.0, xmax)
			flipped_xmax = tf.subtract(1.0, xmin)
			flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], -1)
			return flipped_boxes

		def _flip_image(image):
			image_flipped = tf.image.flip_left_right(image)
			return image_flipped

		do_flip = tf.random.uniform([])
		do_flip = tf.greater(do_flip, 0.5)

		image = tf.cond(do_flip, lambda: _flip_image(image), lambda: image)
		boxes = tf.cond(do_flip, lambda: _flip_boxes(boxes), lambda: boxes)

		return image, classes, boxes

	def _decode_and_preprocess(self, value):
		'''
		Parse and preprocess a single example from a tfrecord.

		Args:
			- value: The value output of a TFRecordReader() object

		Returns:
			- image: The input image.
			- classes: The class ids of the objects in the image (one hot encoded). 
			- bboxes: The bounding boxes coordinates of the objects in the image.
		'''
		features = (
			tf.io.parse_single_example(
				value,
				features={
					'image/encoded': tf.io.FixedLenFeature([], tf.string),
					'image/width': tf.io.FixedLenFeature([], tf.int64),
					'image/height': tf.io.FixedLenFeature([], tf.int64),
					'label/ids': tf.io.VarLenFeature(tf.int64),
					'label/x_mins': tf.io.VarLenFeature(tf.float32),
					'label/y_mins': tf.io.VarLenFeature(tf.float32),
					'label/x_maxs': tf.io.VarLenFeature(tf.float32),
					'label/y_maxs': tf.io.VarLenFeature(tf.float32)
				}
			)
		)

		image_width_original = tf.cast(features['image/width'], tf.int32)
		image_height_original = tf.cast(features['image/height'], tf.int32)

		image = tf.io.decode_image(features['image/encoded'], dtype=tf.float32)
		image = tf.reshape(image, [image_height_original, image_width_original, 3])
		image = self._pad_or_clip(image, self.image_shape)

		classes = tf.one_hot(features['label/ids'].values + 1, self.num_classes + 1)
		classes = self._pad_or_clip(classes, [self.max_num_objects, self.num_classes + 1])
		
		x_mins = features['label/x_mins'].values / tf.cast(self.image_shape[1], tf.float32)
		y_mins = features['label/y_mins'].values / tf.cast(self.image_shape[0], tf.float32)
		x_maxs = features['label/x_maxs'].values / tf.cast(self.image_shape[1], tf.float32)
		y_maxs = features['label/y_maxs'].values / tf.cast(self.image_shape[0], tf.float32)

		bboxes = tf.transpose(tf.stack([x_mins, y_mins, x_maxs, y_maxs]))
		bboxes = self._pad_or_clip(bboxes, [self.max_num_objects, 4])

		return image, classes, bboxes

	def _pad_or_clip(self, tensor, output_shape):
		'''
		Pad or Clip given tensor to the output shape.
		
		Args:
			- tensor: Input tensor to pad or clip.
			- output_shape: A list of integers / scalar tensors (or None for dynamic dim)
			representing the size to pad or clip each dimension of the input tensor.
		
		Returns:
			Input tensor padded and clipped to the output shape.
		'''
		tensor_shape = tf.shape(tensor)
		clip_size = [
			tf.where(tensor_shape[i] - shape > 0, shape, -1)
			if shape is not None else -1 for i, shape in enumerate(output_shape)
		]
		clipped_tensor = tf.slice(
			tensor,
			begin=tf.zeros(len(clip_size), dtype=tf.int32),
			size=clip_size)

		# Pad tensor if the shape of clipped tensor is smaller than the expected
		# shape.
		clipped_tensor_shape = tf.shape(clipped_tensor)
		trailing_paddings = [
			shape - clipped_tensor_shape[i] if shape is not None else 0
			for i, shape in enumerate(output_shape)
		]
		paddings = tf.stack(
			[
				tf.zeros(len(trailing_paddings), dtype=tf.int32),
				trailing_paddings
			],
			axis=1)
		padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
		output_static_shape = [
			dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
		]
		padded_tensor.set_shape(output_static_shape)
		return padded_tensor

