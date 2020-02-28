import tensorflow as tf

import utils.boxes as box_utils
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
		self._image_shape = image_shape
		self._num_classes = num_classes

		self._min_f, self._max_f = foreground_iou_interval
		self._min_b, self._max_b = background_iou_interval

	def generate_targets_batch(self, gt_class_labels, gt_boxes, regions):

		return tf.map_fn(
			fn=lambda x: self._generate_targets(x[0], x[1], regions),
			elems=(gt_class_labels, gt_boxes),
			dtype=(tf.float32, tf.float32))

	def _generate_targets(self, gt_class_labels, gt_boxes, regions):
		'''
		Args:
			- gt_class_labels: A tensor of shape [max_num_objects, num_classes + 1] representing the 
				ground-truth class labels possibly passed with zeros.
			- gt_boxes: A tensor of shape [max_num_objects, 4] representing the 
				ground-truth bounding boxes possibly passed with zeros.
			- regions: A tensor of shape [num_regions, 4] representing the reference regions.
				This corresponds to the anchors for the RPN and to the RoIs for the
				Faster-RCNN.

		Returns:
			- target_class_labels: A tensor of shape [num_regions, num_classes + 1] representing the target
				labels for each region. The target label for ignored regions is zeros(num_classes + 1).
			- target_boxes_encoded: A tensor of shape [num_regions, 4] representing the encoded target ground-truth 
				bounding box for each region, i.e., the ground-truth bounding box with the highest IoU 
				overlap with the considered region. 
		'''

		# Remove padding gt_boxes
		non_null_gt_boxes_inds = tf.reshape(tf.where(
			tf.reduce_sum(gt_class_labels, -1) != 0.0), [-1])
		gt_boxes = tf.gather(gt_boxes, non_null_gt_boxes_inds)

		# Rescale gt_boxes to be in absolute coordnates
		abs_gt_boxes = box_utils.to_absolute(gt_boxes, self._image_shape)

		# Compute pairwise IoU overlap between regions and ground-truth boxes
		ious = metrics.iou(regions, abs_gt_boxes, pairwise=True)
		max_iou_indices = tf.math.argmax(ious, -1)
		max_iou_per_region = tf.reduce_max(ious, axis=-1)
		max_iou = tf.reduce_max(max_iou_per_region)

		# Create target class labels
		background_label = tf.one_hot(0, self._num_classes + 1)
		ignore_label = tf.zeros(self._num_classes + 1)

		background_condition, ignore_condition = self._get_conditions(max_iou_per_region, max_iou)

		target_class_labels = tf.gather(gt_class_labels, max_iou_indices)
		target_class_labels = tf.where(background_condition, background_label, target_class_labels)
		target_class_labels = tf.where(ignore_condition, ignore_label, target_class_labels)

		# Create target boxes
		target_boxes = tf.gather(abs_gt_boxes, max_iou_indices)
		target_boxes_encoded = box_utils.encode(target_boxes, regions)

		return target_class_labels, target_boxes_encoded

	def _get_conditions(self, max_iou_per_region, max_iou):
		foreground_condition = (max_iou_per_region >= self._min_f) & (max_iou_per_region < self._max_f)
		foreground_condition = foreground_condition | (max_iou_per_region == max_iou)

		background_condition = (max_iou_per_region >= self._min_b) & (max_iou_per_region < self._max_b)
		background_condition = background_condition & (max_iou_per_region != max_iou)

		ignore_condition = tf.math.logical_not(foreground_condition | background_condition)

		background_condition = tf.expand_dims(background_condition, 1)
		background_condition = tf.tile(background_condition, [1, self._num_classes + 1])

		ignore_condition = tf.expand_dims(ignore_condition, 1)
		ignore_condition = tf.tile(ignore_condition, [1, self._num_classes + 1])

		return background_condition, ignore_condition