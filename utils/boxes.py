import tensorflow as tf

def encode(boxes, reference_boxes):
	'''
	Args:
		- boxes: A tensor of shape [d1, ..., dN, num_boxes, 4] representing box
			coordinates encoded as: [x_min, y_min, x_max, y_max].
		- reference_boxes: A tensor representing the reference box coordinates.

	Note: This operation supports broadcasting.

	Returns:
		A tensor of same shape as boxes encoding the boxes as
		tensors t = [tx, ty, tw, th], where:
			- tx = (x - xr) / wr
			- ty = (y - yr) / hr
			- tw = log(w / wr)
			- th = log(h / hr)

		With [xr, yr, wr, hr] the coordinates of the associated reference box.

	Raises:
		ValueError: If boxes and reference_boxes cannot be broadcast together.
	'''

	if tf.rank(boxes) < tf.rank(reference_boxes):
		print(tf.rank(boxes))
		print(tf.rank(reference_boxes))
		raise ValueError('operands could not be broadcast together')
	if boxes.shape[tf.rank(boxes) - tf.rank(reference_boxes):] != reference_boxes.shape:
		print(tf.rank(boxes))
		print(tf.rank(reference_boxes))
		raise ValueError('operands could not be broadcast together')

	# Tile reference boxes to support broadcasting
	tiled_reference_boxes = reference_boxes
	multiples = [1 for _ in range(tf.rank(boxes))]
	for i, size in enumerate(boxes.shape[:tf.rank(boxes) - tf.rank(reference_boxes)]):
		tiled_reference_boxes = tf.expand_dims(tiled_reference_boxes, 0)
		multiples[i] = size
	tiled_reference_boxes = tf.tile(tiled_reference_boxes, multiples)

	# encode boxes as [x_center, y_center, width, height]
	mins, maxs = tf.split(boxes, num_or_size_splits=2, axis=-1)
	centers = (maxs + mins) / 2.0
	sizes = maxs - mins

	# encode anchors as [x_center, y_center, width, height] 
	mins_ref, maxs_ref = tf.split(tiled_reference_boxes, num_or_size_splits=2, axis=-1)
	centers_ref = (maxs_ref + mins_ref) / 2.0
	sizes_ref = maxs_ref - mins_ref

	centers = (centers - centers_ref) / sizes_ref
	sizes = tf.math.log(sizes / sizes_ref)

	return tf.concat([centers, sizes], axis=-1)