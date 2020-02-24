import tensorflow as tf

def generate_anchors(scales, aspect_ratios, grid_shape, stride_shape, base_anchor_shape):
		'''
		Creates the anchor boxes

		Args:
			- scales: Anchors scales.
			- aspect_ratios: Anchors aspect ratios.
			- grid_shape: Shape of the anchors grid, i.e., feature maps grid shape.
			- stride_shape: Shape of a pixel projected in the input image.
			- base_anchor_shape: Shape of the base anchor.

		Returns
			A Tensor of shape [num_anchors, 4] representing the box coordinates of the anchors.
			Nb: num_anchors = grid_shape[0] * grid_shape[1] * len(self.scales) * len(self.aspect_ratios)
		'''

		scales, aspect_ratios = tf.meshgrid(scales, aspect_ratios)
		scales = tf.reshape(scales, [-1])
		aspect_ratios = tf.reshape(aspect_ratios, [-1])

		ratio_sqrts = tf.sqrt(aspect_ratios)
		heights = scales / ratio_sqrts * base_anchor_shape[0]
		widths = scales * ratio_sqrts * base_anchor_shape[1]

		x_centers = tf.range(grid_shape[1], dtype=tf.float32) * stride_shape[1]
		y_centers = tf.range(grid_shape[0], dtype=tf.float32) * stride_shape[0]
		x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

		widths, x_centers = tf.meshgrid(widths, x_centers)
		heights, y_centers = tf.meshgrid(heights, y_centers)

		centers = tf.stack([x_centers, y_centers], axis=2)
		centers = tf.reshape(centers, [-1, 2])

		sizes = tf.stack([widths, heights], axis=2)
		sizes = tf.reshape(sizes, [-1, 2])
		
		return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)