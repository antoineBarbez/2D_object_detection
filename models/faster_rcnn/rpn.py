import tensorflow as tf
import numpy as np

import utils.boxes as box_utils
import utils.metrics as metrics


class RPN(tf.keras.Model):

	def __init__(self,
				 image_shape,
				 window_size=3,
				 scales=[0.5, 1.0, 2.0],
				 aspect_ratios=[0.5, 1.0, 2.0]):
		'''
		Args:
			window_size: (Default: 3) Size of the sliding window.
		'''
		super(RPN, self).__init__()

		self.feature_extractor = tf.keras.applications.ResNet50V2(
			input_shape=image_shape,
			include_top=False,
			weights='imagenet')

		initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
		regularizer = tf.keras.regularizers.l2(0.0005)
		self.intermediate_layer = tf.keras.layers.Conv2D(
			filters=256,
			kernel_size=window_size,
			padding='same',
			activation='relu',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='RPN_intermediate_layer')
		
		num_anchors_per_location = len(scales)*len(aspect_ratios)
		self.cls_layer = tf.keras.layers.Conv2D(
			filters=2*num_anchors_per_location, 
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='RPN_classification_head')
		self.cls_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 2),
			name='RPN_reshape_classification_head')
		self.cls_softmax = tf.keras.layers.Activation('softmax')
		self.reg_layer = tf.keras.layers.Conv2D(
			filters=4*num_anchors_per_location,
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='RPN_regression_head')
		self.reg_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 4),
			name='RPN_reshape_regression_head')

		# Losses
		self.crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		self.huber = tf.keras.losses.Huber(
			delta=1.0,
			reduction=tf.keras.losses.Reduction.NONE)

		self.image_shape = image_shape
		self.anchors = self._generate_anchors(scales, aspect_ratios)

	def call(self, images, training=False):
		feature_maps = self.feature_extractor(images, training=training)
		
		features = self.intermediate_layer(feature_maps)
		
		cls_output = self.cls_layer(features)
		cls_output = self.cls_reshape(cls_output)
		cls_output = self.cls_softmax(cls_output)

		reg_output = self.reg_layer(features)
		reg_output = self.reg_reshape(reg_output)

		return cls_output, reg_output

	def _generate_anchors(self, scales, aspect_ratios, stride_shape=(32, 32), base_anchor_shape=(160, 160)):
		'''
		Creates the anchor boxes

		Args:
			- scales: Anchors scales
			- aspect_ratios: Anchors aspect ratios 
			- stride_shape: Shape of a pixel projected in the input image.
			- base_anchor_shape: Shape of the base anchor.

		Returns
			A Tensor of shape [num_anchors, 4] representing the box coordinates of the anchors.
			Nb: num_anchors = grid_shape[0] * grid_shape[1] * len(self.scales) * len(self.aspect_ratios)
		'''
		
		_, grid_height, grid_width, _ = self.feature_extractor.output_shape
		grid_shape = (grid_height, grid_width)

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

	def _generate_targets(self, gt_boxes, num_anchors_per_image):
		'''
		Args:
			gt_boxes: Tensor of shape [batch_size, num_objects, 4] representing 
				the normalized groudtruth bounding boxes coordinates. 

		Returns:
			- target_labels: A tensor of shape [batch_size, num_anchors] representing the objectness
				score for each anchor: 
					- score = 1  --> positive anchor
					- score = 0  --> negative anchor
					- score = -1 --> don't care
			- target_boxes: A tensor of shape [batch_size, num_anchor, 4] containing 
				the coordinates of the groundtruth box that has the highest intersection over union 
				with each anchor.
		'''
		
		batch_size, num_objects, _ = gt_boxes.shape
		num_anchors, _ = self.anchors.shape
		image_height, image_width, _ = self.image_shape

		abs_gt_boxes = tf.multiply(gt_boxes, [image_width, image_height, image_width, image_height])
		tiled_gt_boxes = tf.expand_dims(abs_gt_boxes, 1)
		tiled_gt_boxes = tf.tile(tiled_gt_boxes, [1, num_anchors, 1, 1]) # Shape = [batch_size, num_anchors, num_objects, 4]
		
		batched_anchor_boxes = tf.expand_dims(self.anchors, 0)
		batched_anchor_boxes = tf.tile(batched_anchor_boxes, [batch_size, 1, 1]) # Shape = [batch_size, num_anchors, 4]
		tiled_anchor_boxes = tf.expand_dims(batched_anchor_boxes, 2)
		tiled_anchor_boxes = tf.tile(tiled_anchor_boxes, [1, 1, num_objects, 1]) # Shape = [batch_size, num_anchors, num_objects, 4]
		
		ious = metrics.iou(tiled_gt_boxes, tiled_anchor_boxes).numpy() # Shape = [batch_size, num_anchors, num_objects]
		
		# Create target labels
		max_iou_per_anchor = np.amax(ious, axis=-1)
		max_iou = np.amax(max_iou_per_anchor)
		target_labels = - np.ones((batch_size, num_anchors), dtype=int)
		target_labels[np.where(max_iou_per_anchor == max_iou)] = 1
		target_labels[np.where(max_iou_per_anchor >= 0.7)] = 1
		target_labels[np.where(max_iou_per_anchor <= 0.3)] = 0

		# Anchors that overlap with the image boundaries are ignored during training
		'''inds_to_ignore = np.where(
			(batched_anchor_boxes[:, :, 0] >= 0) &
			(batched_anchor_boxes[:, :, 1] >= 0) &
			(batched_anchor_boxes[:, :, 2] < image_width) &
			(batched_anchor_boxes[:, :, 3] < image_height))
		target_labels[inds_to_ignore] = -1'''

		# Keep only num_anchors_per_image anchors
		batch_inds_to_ignore = np.array([], dtype=int)
		anchor_inds_to_ignore = np.array([], dtype=int)
		for i in range(batch_size):
			positive_inds = np.where(target_labels[i]==1.0)[0]
			negative_inds = np.where(target_labels[i]==0.0)[0]

			num_positive_anchors = positive_inds.size
			num_negative_anchors = negative_inds.size

			num_positive_anchors_to_ignore = num_positive_anchors - num_anchors_per_image // 2
			num_negative_anchors_to_ignore = num_negative_anchors - num_anchors_per_image // 2 + min(0, num_positive_anchors_to_ignore)
			num_positive_anchors_to_ignore = max(0, num_positive_anchors_to_ignore)
			num_negative_anchors_to_ignore = max(0, num_negative_anchors_to_ignore)

			positive_inds_to_ignore = np.random.choice(positive_inds, num_positive_anchors_to_ignore, replace=False)
			negative_inds_to_ignore = np.random.choice(negative_inds, num_negative_anchors_to_ignore, replace=False)
			
			batch_inds_to_ignore = np.concatenate((batch_inds_to_ignore, i * np.ones(num_positive_anchors_to_ignore + num_negative_anchors_to_ignore, dtype=int)))
			anchor_inds_to_ignore = np.concatenate((anchor_inds_to_ignore, positive_inds_to_ignore))
			anchor_inds_to_ignore = np.concatenate((anchor_inds_to_ignore, negative_inds_to_ignore))
		
		inds_to_ignore = (batch_inds_to_ignore, anchor_inds_to_ignore)
		target_labels[inds_to_ignore] = -1
			
		# Create target boxes
		max_iou_indices = tf.expand_dims(tf.math.argmax(ious, -1), -1)
		target_boxes = tf.gather_nd(tiled_gt_boxes, max_iou_indices, batch_dims=2) # Shape = [batch_size, num_anchors, 4]
		target_boxes = box_utils.encode(target_boxes, self.anchors)

		return target_labels, target_boxes

	def get_losses(self, gt_boxes, objectness_pred, boxes_pred, num_anchors_per_image=256):
		'''
		Compute the RPN's classification and regression losses.

		Args:
			- gt_boxes: Tensor of shape [batch_size, num_objects, 4] representing 
				the NORMALIZED groudtruth bounding boxes coordinates.
			- pr_objectness: The output of the RPN's classification head, i.e., 
				a Tensor of shape [batch_size, num_anchors, 2] representing the 
				predicted objectness for each anchor.
			- pr_boxes: The output of the RPN's regression head, i.e., a Tensor
				of shape [batch_size, num_anchors, 4] representing the predicted
				bounding box for each anchor.

		Returns:
			- cls_loss: The value of the classification loss normalized by the batch size.
			- reg_loss: The value of the regression loss normalized by the batch size.
		'''

		# Resize bounding boxes to be in absolute coordinates
		target_labels, target_boxes = tf.py_function(
			self._generate_targets,
			[gt_boxes, num_anchors_per_image],
			[tf.int32, tf.float32])

		# Classification loss
		objectness_inds_to_keep = tf.where(target_labels != -1)
		objectness_true = tf.reshape(tf.gather_nd(target_labels, objectness_inds_to_keep), [-1, num_anchors_per_image])
		objectness_true = tf.one_hot(objectness_true, 2)
		objectness_pred = tf.reshape(tf.gather_nd(objectness_pred, objectness_inds_to_keep), [-1, num_anchors_per_image, 2])
		
		cls_loss = self.crossentropy(objectness_true, objectness_pred)
		
		# Regression loss
		reg_loss = self.huber(target_boxes, boxes_pred)
		reg_loss = tf.reduce_sum(reg_loss, -1)
		reg_loss = tf.gather_nd(reg_loss, tf.where(target_labels == 1))
		reg_loss = tf.reduce_mean(reg_loss)

		return cls_loss, reg_loss
	