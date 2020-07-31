import tensorflow as tf

import utils.boxes as box_utils
import utils.metrics as metrics 

class TargetGenerator(object):
	def __init__(self, image_shape, foreground_iou_interval, background_iou_interval):
		'''
		Args:
			- image_shape: Shape of the input images.
			- foreground_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as foreground. Note that if there are no regions
				within this interval, the region with the highest IoU with any ground truth box will be labeled
				as foreground to ensure that there is always at least one positive example per image.
			- background_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as background. Regions that are neither labeled as
				foreground, nor background are ignored. 
		'''
		super(TargetGenerator, self).__init__()
		self._image_shape = image_shape

		self._min_f, self._max_f = foreground_iou_interval
		self._min_b, self._max_b = background_iou_interval

	def generate_targets(self, gt_labels, gt_boxes, regions):
		'''
		Args:
			- gt_labels: A tensor of shape [max_num_objects, num_classes + 1] representing the 
				ground-truth class labels possibly passed with zeros.
			- gt_boxes: A tensor of shape [max_num_objects, 4] representing the 
				ground-truth bounding boxes possibly passed with zeros.
			- regions: A tensor of shape [num_regions, 4] representing the reference regions.
				This corresponds to the anchors for the RPN and to the RoIs for Fast-RCNN.

		Returns:
			- target_class_labels: A tensor of shape [num_regions, num_classes + 1] representing the one-hot encoded
				target labels for each region. The target label for background regions is [1, 0, ..., 0], whereas the 
				target label for ignored regions is [0, ..., 0]. 
			- target_boxes_encoded: A tensor of shape [num_regions, num_classes, 4] representing the
				encoded target bounding boxes for each region. The values of this tensor are all zeros, 
				except at positions [i, j] such as:
					(1) the i'th region is labeled as a foreground region.
					(2) the object associated with the ground-truth box with the highest IoU with the i'th region
						(let's call it object k) is of the j'th class (without background).
				in this case, target_boxes_encoded[i, j] = gt_boxes[k]
		'''
		num_classes = tf.shape(gt_labels)[1] - 1
		num_regions = tf.shape(regions)[0]

		# Remove padding gt_boxes & gt_class_labels
		non_padding_inds = tf.where(tf.reduce_sum(gt_labels, -1) != 0.0)
		gt_boxes = tf.gather_nd(gt_boxes, non_padding_inds)
		gt_labels = tf.gather_nd(gt_labels, non_padding_inds)

		# Rescale gt_boxes to be in absolute coordnates
		abs_gt_boxes = box_utils.to_absolute(gt_boxes, self._image_shape)

		# Compute pairwise IoU overlap between regions and ground-truth boxes
		ious = metrics.iou(regions, abs_gt_boxes, pairwise=True)
		max_iou_indices = tf.math.argmax(ious, axis=-1)
		max_iou_per_region = tf.reduce_max(ious, axis=-1)
		max_iou = tf.reduce_max(max_iou_per_region)

		# Create target class labels
		background_labels_mask, foreground_labels_mask = self._get_labels_masks(max_iou_per_region, max_iou, num_classes)
		background_labels = tf.one_hot(tf.zeros(num_regions, dtype=tf.int32), num_classes + 1)
		foreground_labels = tf.gather(gt_labels, max_iou_indices)
		
		target_labels = tf.zeros([num_regions, num_classes + 1])
		target_labels = tf.where(background_labels_mask, background_labels, target_labels)
		target_labels = tf.where(foreground_labels_mask, foreground_labels, target_labels)

		# Create target boxes
		foreground_boxes = tf.gather(abs_gt_boxes, max_iou_indices)
		foreground_boxes_encoded = box_utils.encode(foreground_boxes, regions)
		foreground_boxes_encoded = tf.tile(tf.expand_dims(foreground_boxes_encoded, 1), [1, num_classes, 1])
		foreground_boxes_mask = tf.cast(target_labels[:, 1:], dtype=tf.bool)
		foreground_boxes_mask = tf.tile(tf.expand_dims(foreground_boxes_mask, -1), [1, 1, 4])

		target_boxes_encoded = tf.zeros([num_regions, num_classes, 4])
		target_boxes_encoded = tf.where(foreground_boxes_mask, foreground_boxes_encoded, target_boxes_encoded)

		return target_labels, target_boxes_encoded

	def _get_labels_masks(self, max_iou_per_region, max_iou, num_classes):
		"""
		Get background and forerground masks to be used to generate target labels.
		"""
		background_mask = (max_iou_per_region >= self._min_b) & (max_iou_per_region < self._max_b)
		background_mask = tf.tile(tf.expand_dims(background_mask, 1), [1, num_classes + 1])

		max_iou_indice = tf.where(max_iou_per_region == max_iou)[0][0]
		force_one_foreground_mask = tf.one_hot(max_iou_indice, tf.shape(background_mask)[0], on_value=True, off_value=False)
		foreground_mask = (max_iou_per_region >= self._min_f) & (max_iou_per_region < self._max_f)
		foreground_mask = foreground_mask | force_one_foreground_mask
		foreground_mask = tf.tile(tf.expand_dims(foreground_mask, 1), [1, num_classes + 1])

		return background_mask, foreground_mask
		