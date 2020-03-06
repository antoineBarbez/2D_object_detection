import tensorflow as tf

class InputPipelineCreator(object):
	def __init__(self, num_classes, image_shape, max_num_objects):
		'''
		InputPipelineCreator constructor.

		Args:
			- num_classes: Number of classes.
			- image_shape: Shape of the input images. Images that have a different shape
				in the data will be clipped and/or padded to the desired shape. 
			- max_num_objects: Maximum number of object to keep per image.
		'''
		self.num_classes = num_classes
		self.image_shape = image_shape
		self.max_num_objects = max_num_objects
		
	def create_input_pipeline(self, filenames, batch_size=1, training=False):
		'''
		Create an optimized input pipeline.

		Args:
			- filenames: List of paths to TFRecord data files.
			- batch_size: Size of the batches of data.
			- training: Boolean value.

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
		
		if training:
			dataset = dataset.repeat().shuffle(1024).batch(batch_size)
			dataset = dataset.map(
				self._augment_batch,
				num_parallel_calls=tf.data.experimental.AUTOTUNE
			)
		else:
			dataset = dataset.cache()
			dataset = dataset.batch(batch_size)

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
			x_mins, y_mins, x_maxs, y_maxs = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
			flipped_x_mins = tf.subtract(1.0, x_maxs)
			flipped_x_maxs = tf.subtract(1.0, x_mins)
			flipped_boxes = tf.concat([flipped_x_mins, y_mins, flipped_x_maxs, y_maxs], -1)
			return flipped_boxes

		def _flip_image(image):
			image_flipped = tf.image.flip_left_right(image)
			return image_flipped

		do_flip = tf.random.uniform([])
		do_flip = tf.greater(do_flip, 0.5)

		image = tf.cond(do_flip, lambda: _flip_image(image), lambda: image)
		boxes = tf.cond(do_flip, lambda: _flip_boxes(boxes), lambda: boxes)

		return image, classes, boxes

	def _augment_batch(self, images, classes, boxes):
		return tf.map_fn(
			fn=lambda x: self._augment(x[0], x[1], x[2]),
			elems=(images, classes, boxes),
			dtype=(tf.float32, tf.float32, tf.float32))

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

		image = tf.io.decode_image(features['image/encoded'])
		image = tf.cast(image, dtype=tf.float32)
		image = tf.reshape(image, [image_height_original, image_width_original, 3])

		new_height, new_width, _ = self.image_shape
		image = tf.image.resize(image, [new_height, new_width])

		classes = tf.one_hot(features['label/ids'].values + 1, self.num_classes + 1)
		classes = self._pad_or_clip(classes, [self.max_num_objects, self.num_classes + 1])
		
		x_mins = features['label/x_mins'].values / tf.cast(image_width_original, tf.float32)
		y_mins = features['label/y_mins'].values / tf.cast(image_height_original, tf.float32)
		x_maxs = features['label/x_maxs'].values / tf.cast(image_width_original, tf.float32)
		y_maxs = features['label/y_maxs'].values / tf.cast(image_height_original, tf.float32)

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

