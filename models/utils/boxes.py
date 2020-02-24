import tensorflow as tf

def clip_to_window(boxes, window):
	'''
	Args:
		- boxes: Tensor of shape [d1, ..., dN, 4] representing box coordinates.
		- window: Tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
			window to which the op should clip boxes.
	Returns:
		Tensor of shape as boxes.
	'''
	win_x_min, win_y_min, win_x_max, win_y_max = tf.unstack(tf.cast(window, dtype=tf.float32))
	window_max_boudaries = tf.stack(
		[win_x_max, win_y_max, win_x_max, win_y_max])
	window_min_boudaries = tf.stack(
		[win_x_min, win_y_min, win_x_min, win_y_min])

	return tf.maximum(tf.minimum(boxes, window_max_boudaries), window_min_boudaries)


def decode(boxes, reference_boxes):
	'''
	Reverse encode operation.

	Args:
		- boxes: A tensor of shape [d1, ..., dN, num_boxes, 4] representing box
			coordinates encoded as: [x_min, y_min, x_max, y_max].
		- reference_boxes: A tensor representing the reference box coordinates.

	Note: This operation supports broadcasting.

	Returns:
		A tensor of same shape as boxes.
	'''

	#ref_boxes = tf.broadcast_to(reference_boxes, boxes.shape) 
	ref_boxes = tf.expand_dims(reference_boxes, 0)
	ref_boxes = tf.tile(ref_boxes, [tf.shape(boxes)[0], 1])
	mins_ref, maxs_ref = tf.split(ref_boxes, num_or_size_splits=2, axis=-1)
	centers_ref = (maxs_ref + mins_ref) / 2.0
	sizes_ref = maxs_ref - mins_ref

	centers, sizes = tf.split(boxes, num_or_size_splits=2, axis=-1)
	centers = centers * sizes_ref + centers_ref
	sizes = tf.math.exp(sizes) * sizes_ref

	return tf.concat([centers - .5 * sizes, centers + .5 * sizes], axis=-1)

def encode(boxes, reference_boxes):
	'''
	Args:
		- boxes: A tensor of shape [d1, ..., dN, num_boxes, 4] representing box
			coordinates encoded as: [x_min, y_min, x_max, y_max].
		- reference_boxes: A tensor representing the reference box coordinates.

	Note: This operation supports broadcasting.

	Returns:
		A tensor of same shape as boxes, encoding the boxes as
		tensors t = [tx, ty, tw, th], where:
			- tx = (x - xr) / wr
			- ty = (y - yr) / hr
			- tw = log(w / wr)
			- th = log(h / hr)

		With [xr, yr, wr, hr] the coordinates of the associated reference box.
	'''

	#ref_boxes = tf.broadcast_to(reference_boxes, boxes.shape) 
	#ref_boxes = tf.expand_dims(reference_boxes, 0)
	#ref_boxes = tf.tile(ref_boxes, [tf.shape(boxes)[0], 1]) 
	ref_boxes = reference_boxes
	mins_ref, maxs_ref = tf.split(ref_boxes, num_or_size_splits=2, axis=-1)
	centers_ref = (maxs_ref + mins_ref) / 2.0
	sizes_ref = maxs_ref - mins_ref

	mins, maxs = tf.split(boxes, num_or_size_splits=2, axis=-1)
	centers = (maxs + mins) / 2.0
	sizes = maxs - mins

	centers = (centers - centers_ref) / sizes_ref
	sizes = tf.math.log(sizes / sizes_ref)

	return tf.concat([centers, sizes], axis=-1)


