import tensorflow as tf
import tensorflow.keras.backend as K

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


def iou(boxes_1, boxes_2):
	'''
	Intersection over union metric function. 

	Args:
		boxes_1, boxes_2: Input Tensors of shape (d1, ..., dN, 4) both representing a collection of
			box coordinates encoded as: (x_min, y_min, x_max, y_max).

	Returns:
		An output tensor of shape (d1, ..., dN) representing the values of the intersection over union
		between the boxes in boxes_1 and boxes_2.

	Raises:
		ValueError: if boxes_1, boxes_2 do not have the same shape.
		ValueError: if the shape of boxes_1 and boxes_2 does not end by 4.
	'''
	
	if boxes_1.shape != boxes_2.shape:
		raise ValueError('boxes_1 and boxes_2 must be have the same shape.')

	if (boxes_1.shape[-1] != 4):
		raise ValueError('The last dimention of boxes_1 and boxes_2 must be of size 4.')

	x_mins_1, y_mins_1, x_maxs_1, y_maxs_1 = tf.split(boxes_1, num_or_size_splits=4, axis=-1)
	x_mins_2, y_mins_2, x_maxs_2, y_maxs_2 = tf.split(boxes_2, num_or_size_splits=4, axis=-1)

	diff_widths  = K.minimum(x_maxs_1, x_maxs_2) - K.maximum(x_mins_1, x_mins_2)
	diff_heights = K.minimum(y_maxs_1, y_maxs_2) - K.maximum(y_mins_1, y_mins_2)
	intersections = K.maximum(diff_widths, 0) * K.maximum(diff_heights, 0)
	
	widths_1  = x_maxs_1 - x_mins_1
	heights_1 = y_maxs_1 - y_mins_1
	widths_2  = x_maxs_2 - x_mins_2
	heights_2 = y_maxs_2 - y_mins_2

	area_1 = widths_1 * heights_1
	area_2 = widths_2 * heights_2

	unions = K.maximum(area_1 + area_2 - intersections, 0)

	return tf.squeeze(intersections / (unions + K.epsilon()), [-1])
