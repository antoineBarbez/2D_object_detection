import tensorflow as tf

from abc import abstractmethod
from utils.losses import ClassificationLoss, RegressionLoss


class AbstractDetectionModel(tf.keras.Model):
	'''Abstract base class for object detection'''
	def __init__(self, image_shape, **kwargs):
		'''
		Args:
			- image_shape: Shape of the input images.
		'''
		super(AbstractDetectionModel, self).__init__(**kwargs)
		self._image_shape = image_shape

		self._classification_loss = ClassificationLoss()
		self._regression_loss = RegressionLoss()

	@abstractmethod
	def call(self, images, training, **kwargs):
		'''
		Forward pass.

		Args:
			- images: A batch of input images. A [batch_size, height, width, channels] tensor.
			- training: Boolean value indicating whether the training version of the computation
				graph should be constructed.
			- **kwargs: Additional parameters to be passed to the detector's call() method.

		Returns:
			- regions: A tensor of shape [num_regions, 4] representing the reference box coordinates
				to be used for decoding the predicted boxes.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_regions, num_classes, 4] representing encoded predicted box coordinates. 
		'''
		raise NotImplementedError

	@abstractmethod
	def postprocess_output(self, regions, pred_class_scores, pred_boxes_encoded, **kwargs):
		'''
		Args:
			- regions: A tensor of shape [num_regions, 4] representing the reference box coordinates
				to be used for decoding the predicted boxes.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_regions, num_classes, 4] representing encoded predicted box coordinates.
		'''
		raise NotImplementedError

	@tf.function
	def predict(self, images, training=False, **kwargs):
		'''
		Args:
			- images: A batch of input images. A [batch_size, height, width, channels] tensor.
			- training: Boolean.
			- **kwargs: Additional parameters to be passed to the detector's call() method.
		'''
		image_height, image_width, _ = self._image_shape
		resized_images = tf.image.resize(images, [image_height, image_width])
		regions, pred_class_scores, pred_boxes_encoded = self.call(resized_images, training, **kwargs)

		return self.postprocess_output(regions, pred_class_scores, pred_boxes_encoded, training)

	@abstractmethod
	def train_step(self, images, gt_class_labels, gt_boxes, optimizer, num_samples_per_image, **kwargs):
		'''
		Args:
			- images: Input images. A tensor of shape [batch_size, height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [batch_size, num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
				[batch_size, num_objects, 4].
			- optimizer: A tf.keras.optimizers.Optimizer object.
			- num_samples_per_image: Number of examples (regions) to sample per image.
			- **kwargs: Additional parameters to be passed to the detector's call() method.
		'''
		raise NotImplementedError

	@abstractmethod
	def test_step(self, images, gt_class_labels, gt_boxes, num_samples_per_image, **kwargs):
		'''
		Args:
			- images: Input images. A tensor of shape [batch_size, height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [batch_size, num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
				[batch_size, num_objects, 4].
			- num_samples_per_image: Number of examples (regions) to sample per image.
			- **kwargs: Additional parameters to be passed to the detector's call() method.
		'''
		raise NotImplementedError
