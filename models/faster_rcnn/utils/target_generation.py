import tensorflow as tf
import numpy as np

import utils.metrics as metrics

from models.faster_rcnn.utils.box_encoding import encode

def generate_targets(gt_boxes, anchor_boxes, image_shape, num_anchors_per_image):
		'''
		Args:
			- gt_boxes: Tensor of shape [num_objects, 4] representing 
				the normalized groudtruth bounding boxes coordinates.
			- anchor_boxes: Tensor of shape [num_anchors, 4].
			- image_shape: Shape of the input image.
			- num_anchors_per_image: Number of anchors labeled as positive of negative per image.

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
		
		ious = metrics.iou(tiled_gt_boxes, tiled_anchor_boxes).numpy()

		# Anchors that overlap with the image boundaries are ignored during training
		cross_boundary_anchors_inds = np.where(
			(anchor_boxes[:, 0] >= 0) &
			(anchor_boxes[:, 1] >= 0) &
			(anchor_boxes[:, 2] < image_width) &
			(anchor_boxes[:, 3] < image_height))
		cross_boundary_anchors_mask = np.ones(num_anchors, dtype=bool)
		cross_boundary_anchors_mask[cross_boundary_anchors_inds] = False

		ious[cross_boundary_anchors_inds, :] = 0.0
		max_iou_per_anchor = np.amax(ious, axis=-1)
		max_iou = np.amax(max_iou_per_anchor)
		
		target_labels = - np.ones(num_anchors, dtype=int)
		target_labels[np.where(max_iou_per_anchor >= 0.7)] = 1
		target_labels[np.where((max_iou_per_anchor <= 0.3) & cross_boundary_anchors_mask)] = 0
		target_labels[np.where(max_iou_per_anchor == max_iou)] = 1
		
		# Keep only num_anchors_per_image anchors
		positive_inds = np.where(target_labels==1.0)[0]
		negative_inds = np.where(target_labels==0.0)[0]

		num_positive_anchors = positive_inds.size
		num_negative_anchors = negative_inds.size

		num_positive_anchors_to_ignore = num_positive_anchors - num_anchors_per_image // 2
		num_negative_anchors_to_ignore = num_negative_anchors - num_anchors_per_image // 2 + min(0, num_positive_anchors_to_ignore)
		num_positive_anchors_to_ignore = max(0, num_positive_anchors_to_ignore)
		num_negative_anchors_to_ignore = max(0, num_negative_anchors_to_ignore)

		positive_inds_to_ignore = np.random.choice(positive_inds, num_positive_anchors_to_ignore, replace=False)
		negative_inds_to_ignore = np.random.choice(negative_inds, num_negative_anchors_to_ignore, replace=False)
		
		target_labels[positive_inds_to_ignore] = -1
		target_labels[negative_inds_to_ignore] = -1
			
		# Create target boxes
		max_iou_indices = tf.expand_dims(tf.math.argmax(ious, -1), -1)
		target_boxes = tf.gather_nd(tiled_gt_boxes, max_iou_indices, batch_dims=1)
		target_boxes = encode(target_boxes, anchor_boxes)

		return target_labels, target_boxes