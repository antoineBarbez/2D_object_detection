import tensorflow as tf

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

	ref_boxes = tf.broadcast_to(reference_boxes, boxes.shape) 
	mins_ref, maxs_ref = tf.split(ref_boxes, num_or_size_splits=2, axis=-1)
	centers_ref = (maxs_ref + mins_ref) / 2.0
	sizes_ref = maxs_ref - mins_ref

	mins, maxs = tf.split(boxes, num_or_size_splits=2, axis=-1)
	centers = (maxs + mins) / 2.0
	sizes = maxs - mins

	centers = (centers - centers_ref) / sizes_ref
	sizes = tf.math.log(sizes / sizes_ref)

	return tf.concat([centers, sizes], axis=-1)


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

	ref_boxes = tf.broadcast_to(reference_boxes, boxes.shape) 
	mins_ref, maxs_ref = tf.split(ref_boxes, num_or_size_splits=2, axis=-1)
	centers_ref = (maxs_ref + mins_ref) / 2.0
	sizes_ref = maxs_ref - mins_ref

	centers, sizes = tf.split(boxes, num_or_size_splits=2, axis=-1)
	centers = centers * sizes_ref + centers_ref
	sizes = tf.math.exp(sizes) * sizes_ref

	return tf.concat([centers - .5 * sizes, centers + .5 * sizes], axis=-1)

