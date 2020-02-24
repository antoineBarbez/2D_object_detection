import tensorflow as tf

import models.utils.boxes as box_utils
import utils.metrics as metrics 

class TargetGenerator(object):
	def __init__(self, image_shape, num_classes, foreground_iou_interval, background_iou_interval):
		'''
		Args:
			- image_shape: Shape of the input images.
			- num_classes: Number of classes without background.
			- foreground_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as foreground. 
			- background_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as background.
		'''
		super(TargetGenerator, self).__init__()
		self.image_shape = image_shape
		self.num_classes = num_classes

		self.min_f, self.max_f = foreground_iou_interval
		self.min_b, self.max_b = background_iou_interval

	def generate_targets(self, gt_labels, gt_boxes, regions):
		'''
		Args:
			- gt_labels: A tensor of shape [max_num_objects, num_classes + 1] representing the 
				ground-truth class labels possibly passed with zeros.
			- gt_boxes: A tensor of shape [max_num_objects, 4] representing the 
				ground-truth bounding boxes possibly passed with zeros.
			- regions: A tensor of shape [num_regions, 4] representing the reference regions.
				This corresponds to the anchors for the RPN and to the RoIs for the
				Faster-RCNN.

		Returns:
			- target_labels: A tensor of shape [num_regions, num_classes + 1] representing the target
				labels for each region. The target label for ignored regions is zeros(num_classes + 1).
			- target_boxes: A tensor of shape [num_regions, 4] representing the target ground-truth bounding box
				for each region, i.e., the ground-truth bounding box with the higher IoU overlap with the
				considered region. 
		'''

		image_height, image_width, _ = self.image_shape
		abs_gt_boxes = tf.multiply(gt_boxes, [image_width, image_height, image_width, image_height])
		
		# TODO
		# Remove padded gt_boxes

		ious = metrics.iou(regions, abs_gt_boxes, pairwise=True)
		max_iou_indices = tf.math.argmax(ious, -1)
		max_iou_per_region = tf.reduce_max(ious, axis=-1)
		max_iou = tf.reduce_max(max_iou_per_region)

		# Create target labels
		background_label = tf.one_hot(0, self.num_classes + 1)
		ignore_label = tf.zeros(self.num_classes + 1)

		background_condition, ignore_condition = self._get_conditions(max_iou_per_region, max_iou)

		target_labels = tf.gather(gt_labels, max_iou_indices)
		target_labels = tf.where(background_condition, background_label, target_labels)
		target_labels = tf.where(ignore_condition, ignore_label, target_labels)

		# Create target boxes
		target_boxes = tf.gather(abs_gt_boxes, max_iou_indices)
		target_boxes = box_utils.encode(target_boxes, regions)

		return target_labels, target_boxes

	def _get_conditions(self, max_iou_per_region, max_iou):
		foreground_condition = (max_iou_per_region >= self.min_f) & (max_iou_per_region < self.max_f)
		foreground_condition = foreground_condition | (max_iou_per_region == max_iou)

		background_condition = (max_iou_per_region >= self.min_b) & (max_iou_per_region < self.max_b)
		background_condition = background_condition & (max_iou_per_region != max_iou)

		ignore_condition = tf.math.logical_not(foreground_condition | background_condition)

		background_condition = tf.expand_dims(background_condition, 1)
		background_condition = tf.tile(background_condition, [1, self.num_classes + 1])

		ignore_condition = tf.expand_dims(ignore_condition, 1)
		ignore_condition = tf.tile(ignore_condition, [1, self.num_classes + 1])

		return background_condition, ignore_condition