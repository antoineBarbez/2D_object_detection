import tensorflow as tf
import numpy as np

import utils.metrics as metrics

from models.utils.box_encoding import encode

def generate_targets(gt_boxes, anchor_boxes, image_shape, num_anchors_to_keep):
		'''
		Args:
			- gt_boxes: Tensor of shape [num_objects, 4] representing 
				the groudtruth bounding boxes in relative coordinates.
			- anchor_boxes: Tensor of shape [num_anchors, 4] representing the anchor
				boxes in absolute coordinates.
			- image_shape: Shape of the input image.
			- num_anchors_to_keep: Number of anchors labeled as positive of negative per image.

		Returns:
			- target_labels: A tensor of shape [num_anchors] representing the objectness
				score for each anchor: 
					- score = 1  --> positive anchor
					- score = 0  --> negative anchor
					- score = -1 --> don't care
			- target_boxes: A tensor of shape [num_anchors, 4] containing 
				the coordinates of the groundtruth box that has the highest intersection over union 
				with each anchor.
		'''
		
		num_objects = gt_boxes.shape[0]
		num_anchors = anchor_boxes.shape[0]
		image_height, image_width, _ = image_shape

		abs_gt_boxes = tf.multiply(gt_boxes, [image_width, image_height, image_width, image_height])
		tiled_gt_boxes = tf.expand_dims(abs_gt_boxes, 0)
		tiled_gt_boxes = tf.tile(tiled_gt_boxes, [num_anchors, 1, 1])
		
		tiled_anchor_boxes = tf.expand_dims(anchor_boxes, 1)
		tiled_anchor_boxes = tf.tile(tiled_anchor_boxes, [1, num_objects, 1])
		
		ious = metrics.iou(tiled_gt_boxes, tiled_anchor_boxes)

		# Anchors that overlap with the image boundaries are ignored during training
		cross_boundary_anchors_mask = tf.where(
			(anchor_boxes[:, 0] >= 0) &
			(anchor_boxes[:, 1] >= 0) &
			(anchor_boxes[:, 2] < image_width) &
			(anchor_boxes[:, 3] < image_height), False, True)
		
		# Set IOU to zero for cross-boundary anchors
		mask_iou = tf.tile(tf.expand_dims(cross_boundary_anchors_mask, -1), [1, num_objects])
		ious = tf.where(mask_iou, ious, tf.zeros([num_anchors, num_objects]))

		# Create target boxes
		max_iou_indices = tf.expand_dims(tf.math.argmax(ious, -1), -1)
		target_boxes = tf.gather_nd(tiled_gt_boxes, max_iou_indices, batch_dims=1)
		target_boxes = encode(target_boxes, anchor_boxes)

		# Creat target labels
		max_iou_per_anchor = tf.reduce_max(ious, axis=-1)
		max_iou = tf.reduce_max(max_iou_per_anchor)

		positive_inds = tf.where(
			(max_iou_per_anchor >= 0.7) |
			(max_iou_per_anchor == max_iou))
		
		negative_inds = tf.where(
			(max_iou_per_anchor <= 0.3) &
			(max_iou_per_anchor != max_iou) &
			cross_boundary_anchors_mask)

		positive_inds = tf.random.shuffle(positive_inds)
		negative_inds = tf.random.shuffle(negative_inds)

		num_positive_anchors = tf.size(positive_inds)
		num_negative_anchors = tf.size(negative_inds)

		num_positive_anchors_to_keep = tf.minimum(num_positive_anchors, num_anchors_to_keep // 2)
		num_negative_anchors_to_keep = num_anchors_to_keep - num_positive_anchors_to_keep

		positive_inds = positive_inds[:num_positive_anchors_to_keep]
		negative_inds = negative_inds[:num_negative_anchors_to_keep]
		
		target_labels = tf.sparse.SparseTensor(
			indices=tf.concat([
				positive_inds,
				negative_inds], axis=0),
			values=tf.concat([
				tf.ones(num_positive_anchors_to_keep, dtype=tf.int32),
				tf.zeros(num_negative_anchors_to_keep, dtype=tf.int32)], axis=0),
			dense_shape=[num_anchors])
		target_labels = tf.sparse.reorder(target_labels)
		target_labels = tf.sparse.to_dense(target_labels, default_value=-1)

		return target_boxes, target_labels