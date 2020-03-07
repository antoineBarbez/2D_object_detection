import tensorflow as tf
import utils.boxes as box_utils
import utils.anchors as anchor_utils

from models.object_detection_model import ObjectDetectionModel
from models.detectors.rpn_detector import RPNDetector

class RPN(ObjectDetectionModel):
	def __init__(self,
		image_shape,
		window_size=3,
		scales=[0.25, 0.5, 1.0, 2.0],
		aspect_ratios=[0.5, 1.0, 2.0],
		base_anchor_shape=(160, 160),
		name='region_proposal_network'):
		'''
		Instantiate a Region Proposal Network.

		Args:
			- image_shape: Shape of the input images.
			- num_classes: Number of classes without background.
			- window_size: (Default: 3) Size of the sliding window.
			- scales: Anchors' scales.
			- aspect_ratios: Anchors' aspect ratios.
			- base_anchor_shape: Shape of the base anchor. 
		'''
		super(RPN, self).__init__(
			image_shape=image_shape,
			num_classes=1,
			foreground_proportion=0.5,
			foreground_iou_interval=(0.65, 1.0), 
			background_iou_interval=(0.0, 0.3),
			name=name)

		_, grid_height, grid_width, _ = self._feature_extractor.output_shape
		grid_shape = (grid_height, grid_width)

		self._detector = RPNDetector(
			image_shape=image_shape,
			grid_shape=grid_shape ,
			window_size=window_size,
			scales=scales,
			aspect_ratios=aspect_ratios,
			base_anchor_shape=base_anchor_shape)

	# Overrides ObjectDetectionModel.train_step
	def train_step(self, images, gt_class_labels, gt_boxes, optimizer, num_samples_per_image=256):
		gt_objectness_labels = tf.one_hot(
			indices=tf.cast(
				tf.reduce_sum(gt_class_labels, -1),
				dtype=tf.int32),
			depth=2)

		return super(RPN, self).train_step(
			images=images,
			gt_class_labels=gt_objectness_labels,
			gt_boxes=gt_boxes,
			optimizer=optimizer,
			num_samples_per_image=num_samples_per_image)

	# Overrides ObjectDetectionModel.test_step
	def test_step(self, images, gt_class_labels, gt_boxes, num_samples_per_image=256):
		gt_objectness_labels = tf.one_hot(
			indices=tf.cast(
				tf.reduce_sum(gt_class_labels, -1),
				dtype=tf.int32),
			depth=2)

		return super(RPN, self).test_step(
			images=images,
			gt_class_labels=gt_objectness_labels,
			gt_boxes=gt_boxes,
			num_samples_per_image=num_samples_per_image)
	
