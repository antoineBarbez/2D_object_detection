import tensorflow as tf
import utils.anchors as anchor_utils
import utils.boxes as box_utils
import utils.post_processing as postprocess_utils

from models.detectors.abstract_detector import AbstractDetector

class RPNDetector(AbstractDetector):
	def __init__(self,
		image_shape,
		grid_shape,
		window_size,
		scales,
		aspect_ratios,
		base_anchor_shape,
		name='region_proposal_network_detector'):
		'''
		Instantiate a Region Proposal Network.

		Args:
			- image_shape: Shape of the input images.
			- grid_shape: Tuple of integers. Shape of the anchors grid, 
				i.e., height and width of the feature maps.
			- window_size: Size of the sliding window.
			- scales: Anchors' scales.
			- aspect_ratios: Anchors' aspect ratios.
			- base_anchor_shape: Tuple of integers. Shape of the base anchor. 
		'''
		super(RPNDetector, self).__init__(name=name)

		initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
		regularizer = tf.keras.regularizers.l2(0.0005)
		self._intermediate_layer = tf.keras.layers.Conv2D(
			filters=256,
			kernel_size=window_size,
			padding='same',
			activation='relu',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_intermediate_layer')
		
		num_anchors_per_location = len(scales)*len(aspect_ratios)
		self._cls_layer = tf.keras.layers.Conv2D(
			filters=2*num_anchors_per_location, 
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_classification_head')
		self._cls_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 2),
			name='rpn_classification_head_reshape')
		self._cls_activation = tf.keras.layers.Activation(
			activation='softmax',
			name='rpn_classification_head_activation')
		
		self._reg_layer = tf.keras.layers.Conv2D(
			filters=4*num_anchors_per_location,
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_regression_head')
		self._reg_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 1, 4),
			name='rpn_regression_head_reshape')

		self._image_shape = image_shape
		self._anchors = anchor_utils.generate_anchors(
			scales=scales,
			aspect_ratios=aspect_ratios,
			grid_shape=grid_shape,
			stride_shape=(16, 16), 
			base_anchor_shape=base_anchor_shape)

	@property
	def anchors(self):
		return self._anchors

	def call(self, feature_maps, training):
		'''
		Args:
			- feature_maps: Output of the feature extractor.
			- training: A boolean indicating whether the training version of the
				computation graph should be constructed.

		Returns:
			- anchors: Anchor boxes to be used for postprocessing, shape = [num_anchors, 4].
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_anchors, 2] representing classification scores for each anchor.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape 
				[batch_size, num_anchors, 1, 4] representing encoded predicted box 
				coordinates for each anchor.
		'''
		features = self._intermediate_layer(feature_maps)
		
		pred_class_scores = self._cls_layer(features)
		pred_class_scores = self._cls_reshape(pred_class_scores)
		pred_class_scores = self._cls_activation(pred_class_scores)

		pred_boxes_encoded = self._reg_layer(features)
		pred_boxes_encoded = self._reg_reshape(pred_boxes_encoded)

		if training:
			anchors, pred_class_scores, pred_boxes_encoded = (
				self._remove_invalid_anchors_and_predictions(pred_class_scores, pred_boxes_encoded))
		else:
			image_height, image_width, _ = self._image_shape
			anchors = box_utils.clip_to_window(self._anchors, [0, 0, image_width, image_height])
			
		return anchors, pred_class_scores, pred_boxes_encoded
	
	def postprocess_output(self, anchors, pred_class_scores, pred_boxes_encoded, training):
		'''
		Postprocess the output of the RPN

		Args:
			- anchors: A tensor of shape [num_anchors, 4] representing the anchor boxes.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_anchors, 2] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_anchors, 1, 4] representing encoded predicted box coordinates.
			- training: A boolean value indicating whether we are in training mode.

		Returns:
			- rois: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded 
				representing region proposals.
			- roi_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded 
				representing objectness scores for each proposal.
		'''
		max_predictions = 2000 if training else 300

		nmsed_boxes, nmsed_scores, _, _ = postprocess_utils.postprocess_output(
				regions=anchors,
				image_shape=self._image_shape,
				pred_class_scores=pred_class_scores,
				pred_boxes_encoded=pred_boxes_encoded,
				max_output_size_per_class=max_predictions,
				max_total_size=max_predictions,
				iou_threshold=0.7)

		return nmsed_boxes, nmsed_scores

	def _remove_invalid_anchors_and_predictions(self, pred_class_scores, pred_boxes_encoded):
		'''
		Remove anchors that overlap with the image boundaries, as well as 
		the corresponding predictions.

		Args:
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_anchors, 2] representing classification scores for each anchor.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape 
				[batch_size, num_anchors, 1, 4] representing encoded predicted box coordinates 
				for each anchor.
		
		Returns:
			filtered anchors, pred_class_scores, and pred_boxes_encoded.
		'''
		image_height, image_width, _ = self._image_shape
		inds_to_keep = tf.reshape(
			tf.where(
				(self.anchors[:, 0] >= 0) &
				(self.anchors[:, 1] >= 0) &
				(self.anchors[:, 2] <= image_width) &
				(self.anchors[:, 3] <= image_height)),
			[-1])

		anchors = tf.gather(self._anchors, inds_to_keep)
		pred_class_scores = tf.gather(pred_class_scores, inds_to_keep, axis=1)
		pred_boxes_encoded = tf.gather(pred_boxes_encoded, inds_to_keep, axis=1)

		return anchors, pred_class_scores, pred_boxes_encoded