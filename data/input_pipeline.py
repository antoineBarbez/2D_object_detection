import tensorflow as tf

class InputPipelineCreator(object):
	def __init__(self, num_classes, image_shape):
		'''
		InputPipelineCreator constructor

		Args:
			- num_classes: Number of classes.
			- image_shape: Shape of the input images. Images that have a different shape
				in the data will be clipped and/or padded to the desired shape.
		'''
		self.num_classes = num_classes
		self.image_shape = image_shape
		
	def create_input_pipeline(self, filenames, batch_size, augmentation=False):
		'''
		Create an optimized input pipeline

		Args:
			- filenames: List of paths to TFRecord data files.
			- batch_size: Number of samples per batch.
			- augmentation: (Optional) Whether to apply data augmentation 
				(i.e., random transformations) to the images.

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
		dataset = dataset.shuffle(1024)
		dataset = dataset.padded_batch(
			batch_size=batch_size,
			drop_remainder=True,
			padded_shapes=(
				self.image_shape,
				[None, self.num_classes],
				[None, 4]
			)
		)
		if augmentation:
			dataset = dataset.map(
				self._augment,
				num_parallel_calls=tf.data.experimental.AUTOTUNE
			)
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
		
		return dataset

	# TODO
	def _augment(self, images, classes, bboxes):
		'''
		Apply random transformations to a batch of data

		Args:
			images, classes, bboxes: Batch of images, classes, and bounding boxes.

		Returns:
			A batch of transformed images, classes, and bounding boxes.
		'''
		return images, classes, bboxes

	def _decode_and_preprocess(self, value):
		'''
		Parse and preprocess a single example from a tfrecord

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

		image = tf.io.decode_image(features['image/encoded'], dtype=tf.float32) / 255.0
		image = tf.reshape(image, [image_height_original, image_width_original, 3])
		image = self._pad_or_clip(image, self.image_shape)

		classes = tf.one_hot(features['label/ids'].values, self.num_classes)
		
		x_mins = features['label/x_mins'].values / tf.cast(self.image_shape[1], tf.float32)
		y_mins = features['label/y_mins'].values / tf.cast(self.image_shape[0], tf.float32)
		x_maxs = features['label/x_maxs'].values / tf.cast(self.image_shape[1], tf.float32)
		y_maxs = features['label/y_maxs'].values / tf.cast(self.image_shape[0], tf.float32)

		bboxes = tf.transpose(tf.stack([x_mins, y_mins, x_maxs, y_maxs]))

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

