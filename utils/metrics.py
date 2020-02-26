import tensorflow as tf

class MeanIOU(tf.keras.metrics.Mean):
	'''
	Mean intersection over union metric class
	'''
	def __init__(self, name='intersection_over_union', dtype=None):
		'''
		Creates a `IOU` instance.
		
		Args:
			name: (Optional) string name of the metric instance.
			dtype: (Optional) data type of the metric result.
		'''
		super(MeanIOU, self).__init__(name=name, dtype=dtype)

	def update_state(self, gt_boxes, pr_boxes):
		'''
		Accumulates metric statistics.
		
		Args:
			gt_boxes: Batch of ground truth bounding boxes coordinates
			pr_boxes: Batch of predicted bounding boxes coordinates.
		
		Returns:
			Update op.
		'''
		mean_iou = tf.reduce_mean(iou(gt_boxes, pr_boxes))

		return super(IOU, self).update_state(mean_iou)

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

