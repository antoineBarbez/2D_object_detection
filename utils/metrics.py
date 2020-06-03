import tensorflow as tf

import functools

class AveragePrecision(tf.keras.metrics.Metric):
	def __init__(self, iou_threshold, **kwargs):
		super(AveragePrecision, self).__init__(**kwargs)

		self.iou_threshold = iou_threshold

		self.auc = tf.keras.metrics.AUC(num_thresholds=51, curve='PR')

	def reset_states(self):
		self.auc.reset_states()

	def result(self):
		return self.auc.result()

	def update_state(self, gt_boxes, pred_boxes, pred_scores):
		'''
		Args:
			- gt_boxes: A tensor of shape [batch_size, max_num_objects, 4] possibly zero padded 
				representing the ground-truth boxes.
			- pred_boxes: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded 
				representing predicted bounding-boxes.
			- pred_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded 
				representing the scores for each predicted box.
		'''
		batch_size = tf.shape(pred_boxes)[0]
		for i in range(batch_size):
			self.update_state_single_image(
				gt_boxes=gt_boxes[i],
				pred_boxes=pred_boxes[i],
				pred_scores=pred_scores[i])

	def update_state_single_image(self, gt_boxes, pred_boxes, pred_scores):
		'''
		Args:
			- gt_boxes: A tensor of shape [max_num_objects, 4] possibly zero padded 
				representing the ground-truth boxes.
			- pred_boxes: A tensor of shape [max_predictions, 4] possibly zero padded 
				representing predicted bounding-boxes.
			- pred_scores: A tensor of shape [max_predictions] possibly zero padded 
				representing the scores for each predicted box.
		'''

		# Remove padding boxes and scores
		gt_non_padding_inds = tf.where(tf.reduce_sum(gt_boxes, -1) != 0.0, name='where_ap_1')
		gt_boxes = tf.gather_nd(gt_boxes, gt_non_padding_inds)

		pred_non_padding_inds = tf.where(tf.reduce_sum(pred_boxes, -1) != 0.0, name='where_ap_2')
		pred_boxes = tf.gather_nd(pred_boxes, pred_non_padding_inds)
		pred_scores = tf.gather_nd(pred_scores, pred_non_padding_inds)

		# Compute true scores
		ious = iou(gt_boxes, pred_boxes, pairwise=True)
		ious = tf.reduce_max(ious, axis=0)
		true_scores = tf.cast(ious > self.iou_threshold, dtype=tf.float32)

		self.auc.update_state(true_scores, pred_scores)

class MeanAveragePrecision(tf.keras.metrics.Metric):
	def __init__(self, num_classes, iou_threshold, **kwargs):
		super(MeanAveragePrecision, self).__init__(**kwargs)
		self.num_classes = num_classes 

		self.average_precisions = [AveragePrecision(iou_threshold) for _ in range(num_classes)]

	def reset_states(self):
		for average_precision in self.average_precisions:
			average_precision.reset_states()

	def result(self):
		sum_ap = 0.0
		for average_precision in self.average_precisions:
			sum_ap += average_precision.result()

		return sum_ap / float(self.num_classes)

	def update_state(self, gt_boxes, gt_class_labels, pred_boxes, pred_scores, pred_classes):
		'''
		Args:
			- gt_boxes: A tensor of shape [batch_size, max_num_objects, 4] possibly zero padded 
				representing the ground-truth boxes.
			- gt_class_labels: [batch_size, max_num_objects, num_classes] possibly zero padded 
				representing the ground-truth class labels.
			- pred_boxes: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded 
				representing predicted bounding-boxes.
			- pred_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded 
				representing the class score for each predicted box.
			- pred_classes: A tensor of shape [batch_size, max_predictions] possibly zero padded 
				representing the class indice for each predicted box.
		'''
		for i in range(self.num_classes):
			self.average_precisions[i].update_state(
				gt_boxes=tf.where(
					condition=tf.tile(
						input=tf.expand_dims(gt_class_labels[:, :, i] == 1.0, -1),
						multiples=[1, 1, 4]),
					x=gt_boxes,
					y=0.0),
				pred_boxes=tf.where(
					condition=tf.tile(
						input=tf.expand_dims(pred_classes == i, -1),
						multiples=[1, 1, 4]),
					x=pred_boxes,
					y=0.0),
				pred_scores=tf.where(
					condition=pred_classes == i,
					x=pred_scores,
					y=0.0))

def area(boxes):
	'''
	Computes area of boxes.
	
	Args:
		- boxes: A tf.float32 tensor of shape [N, 4].
	
	Returns:
		A tf.float32 tensor of shape [N].
	'''
	x_mins, y_mins, x_maxs, y_maxs = tf.split(
		boxes, num_or_size_splits=4, axis=-1)
	return tf.squeeze((x_maxs - x_mins) * (y_maxs - y_mins), [-1])

def intersection(boxes_1, boxes_2, pairwise=False):
	'''
	Compute pairwise intersection areas between boxes.
	
	Args:
		- boxes_1: A tf.float32 tensor of shape [N, 4].
		- boxes_2: A tf.float32 tensor of shape [M, 4].
		- pairwise: A boolean, if True this operation returns the pairwise
			intersection values else, it returns the elementwise intersections.

	Returns:
		A tf.float32 tensor of shape [N, M] if pairwise is True
		else, a tf.float32 tensor of shape [N].
	'''
	x_mins_1, y_mins_1, x_maxs_1, y_maxs_1 = tf.split(
		boxes_1, num_or_size_splits=4, axis=-1)
	x_mins_2, y_mins_2, x_maxs_2, y_maxs_2 = tf.split(
		boxes_2, num_or_size_splits=4, axis=-1)

	if pairwise:
		x_mins_2 = tf.transpose(x_mins_2)
		y_mins_2 = tf.transpose(y_mins_2)
		x_maxs_2 = tf.transpose(x_maxs_2)
		y_maxs_2 = tf.transpose(y_maxs_2)

	diff_widths  = tf.minimum(x_maxs_1, x_maxs_2) - tf.maximum(x_mins_1, x_mins_2)
	diff_heights = tf.minimum(y_maxs_1, y_maxs_2) - tf.maximum(y_mins_1, y_mins_2)
	intersections = tf.maximum(0.0, diff_widths) * tf.maximum(0.0, diff_heights)
	
	if pairwise:
		return intersections
	else:
		return tf.reshape(intersections, [-1])

def iou(boxes_1, boxes_2, pairwise=False):
	'''
	Computes intersection-over-union between two set of boxes.
	
	Args:
		- boxes_1: A tf.float32 tensor of shape [N, 4].
		- boxes_2: A tf.float32 tensor of shape [M, 4].
		- pairwise: A boolean, if True this operation returns the pairwise
			iou scores else, it returns the elementwise iou scores.

	Returns:
		A tf.float32 tensor of shape [N, M] if pairwise is True
		else, a tf.float32 tensor of shape [N].
	'''

	intersections = intersection(boxes_1, boxes_2, pairwise)
	
	areas_1 = area(boxes_1)
	areas_2 = area(boxes_2)
	if pairwise:
		areas_1 = tf.expand_dims(areas_1, 1)
		areas_2 = tf.expand_dims(areas_2, 0)
	
	unions = areas_1 + areas_2 - intersections
	
	return tf.where(
		tf.equal(intersections, 0.0),
		tf.zeros_like(intersections), tf.truediv(intersections, unions))

