import tensorflow as tf
import models.utils.boxes as box_utils
import models.utils.anchors as anchor_utils

from models.object_detection_model import ObjectDetectionModel

class RPN(ObjectDetectionModel):
	def __init__(self,
		image_shape,
		window_size=3,
		scales=[0.5, 1.0, 2.0],
		aspect_ratios=[0.5, 1.0, 2.0],
		base_anchor_shape=(160, 160),
		name='region_proposal_network'):
		'''
		Instantiate a Region Proposal Network.

		Args:
			- image_shape: Shape of the input images.
			- num_classes: Number of classes without background.
			- window_size: (Default: 3) Size of the sliding window.
			- scales: Anchors' scales.
			- aspect_ratios: Anchors' aspect ratios.
			- base_anchor_shape: Shape of the base anchor. 
		'''
		super(RPN, self).__init__(
			image_shape=image_shape,
			num_classes=1,
			foreground_proportion=0.5,
			foreground_iou_interval=(0.7, 1.0), 
			background_iou_interval=(0.0, 0.3),
			name=name)

		initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
		regularizer = tf.keras.regularizers.l2(0.0005)
		self.intermediate_layer = tf.keras.layers.Conv2D(
			filters=256,
			kernel_size=window_size,
			padding='same',
			activation='relu',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_intermediate_layer')
		
		num_anchors_per_location = len(scales)*len(aspect_ratios)
		self.cls_layer = tf.keras.layers.Conv2D(
			filters=2*num_anchors_per_location, 
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_classification_head')
		self.cls_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 2),
			name='rpn_classification_head_reshape')
		self.cls_activation = tf.keras.layers.Activation(
			activation='softmax',
			name='rpn_classification_head_activation')
		
		self.reg_layer = tf.keras.layers.Conv2D(
			filters=4*num_anchors_per_location,
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_regression_head')
		self.reg_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 1, 4),
			name='rpn_regression_head_reshape')

		_, grid_height, grid_width, _ = self.feature_extractor.output_shape
		grid_shape = (grid_height, grid_width)
		self.anchors = anchor_utils.generate_anchors(
			scales=scales,
			aspect_ratios=aspect_ratios,
			grid_shape=grid_shape,
			stride_shape=(16, 16), 
			base_anchor_shape=base_anchor_shape)
		self.image_shape = image_shape

	def postprocess_output(self, anchors, cls_output, reg_output, training):
		'''
		Postprocess the output of the RPN

		Args:
			- cls_output: Output of the classification head. A tensor of shape 
				[num_anchors, 2] representing classification scores for each anchor.
			- reg_output: Output of the regression head. A tensor of shape [num_anchors, num_class=1, 4]
				representing encoded predicted box coordinates for each anchor.
			- training: A boolean value.

		Returns:
			roi_scores: A set of objectness scores for the rois.
			rois: A set of regions of interest, i.e., A set of box proposals.
		'''
		boxes = box_utils.decode(reg_output, anchors)

		# Clip boxes to image boundaries
		image_height, image_width, _ = self.image_shape
		boxes = box_utils.clip_to_window(boxes, [0, 0, image_width, image_height])

		# Non Max Suppression
		inds_to_keep = tf.image.non_max_suppression(
		    boxes=boxes,
		    scores=cls_output,
		    max_output_size=2000 if training else 300,
		    iou_threshold=0.7)

		rois = tf.gather(boxes, inds_to_keep)
		roi_scores = tf.gather(cls_output, inds_to_keep)

		return roi_scores, rois

	def _predict(self, feature_maps, training):
		'''
		Args:
			- feature_maps: Output of the feature extractor.
			- training: A boolean indicating whether the training version of the
				computation graph should be constructed.

		Returns:
			- anchors: Anchor boxes to be used for postprocessing, shape = [num_anchors, 4].
			- cls_output: Output of the classification head. A tensor of shape 
				[num_anchors, 2] representing classification scores for each anchor.
			- reg_output: Output of the regression head. A tensor of shape [num_anchors, num_class=1, 4]
				representing encoded predicted box coordinates for each anchor.
		'''
		features = self.intermediate_layer(feature_maps)
		
		cls_output = self.cls_layer(features)
		cls_output = self.cls_reshape(cls_output)
		cls_output = self.cls_activation(cls_output)
		cls_output = tf.squeeze(cls_output, [0])

		reg_output = self.reg_layer(features)
		reg_output = self.reg_reshape(reg_output)
		reg_output = tf.squeeze(reg_output, [0])

		if training:
			anchors, cls_output, reg_output = self._remove_invalid_anchors_and_predictions(cls_output, reg_output)
		else:
			image_height, image_width, _ = self.image_shape
			anchors = box_utils.clip_to_window(self.anchors, [0, 0, image_width, image_height])
			
		return anchors, cls_output, reg_output

	def _remove_invalid_anchors_and_predictions(self, cls_output, reg_output):
		'''
		Remove anchors that overlap with the image boundaries, as well as 
		the corresponding predictions.

		Args:
			- cls_output: Output of the classification head. A tensor of shape 
				[num_anchors, 2] representing classification scores for each anchor.
			- reg_output: Output of the regression head. A tensor of shape [num_anchors, num_class=1, 4]
				representing encoded predicted box coordinates for each anchor.
		
		Returns:
			filtered anchors, cls_output, and reg_output.
		'''
		image_height, image_width, _ = self.image_shape
		inds_to_keep = tf.reshape(
			tf.where(
				(self.anchors[:, 0] >= 0) &
				(self.anchors[:, 1] >= 0) &
				(self.anchors[:, 2] <= image_width) &
				(self.anchors[:, 3] <= image_height)),
			[-1])

		filtered_anchors = tf.gather(self.anchors, inds_to_keep)
		filtered_cls_output = tf.gather(cls_output, inds_to_keep)
		filtered_reg_output = tf.gather(reg_output, inds_to_keep)

		return filtered_anchors, filtered_cls_output, filtered_reg_output
