import tensorflow as tf

from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from models.utils.anchor_generation import generate_anchors
from models.utils.box_encoding import decode
from models.utils.target_generation import generate_targets

class RPN(tf.keras.Model):
	def __init__(self,
		image_shape,
		window_size=3,
		scales=[0.5, 1.0, 2.0],
		aspect_ratios=[0.5, 1.0, 2.0],
		base_anchor_shape=(160, 160),
		name='region_proposal_network',
		**kwargs):
		'''
		Args:
			- image_shape: Shape of the input images.
			- window_size: (Default: 3) Size of the sliding window.
			- scales: Anchors' scales.
			- aspect_ratios: Anchors' aspect ratios.
			- base_anchor_shape: Shape of the base anchor. 
		'''
		super(RPN, self).__init__(name=name, **kwargs)

		self.feature_extractor = ResNet50FeatureExtractor(
			kernel_regularizer=tf.keras.regularizers.l2(0.00005),
			input_shape=image_shape)

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
			filters=num_anchors_per_location, 
			kernel_size=1,
			padding='same',
			activation='sigmoid',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_classification_head')
		self.cls_reshape = tf.keras.layers.Flatten(name='RPN_reshape_classification_head')
		
		self.reg_layer = tf.keras.layers.Conv2D(
			filters=4*num_anchors_per_location,
			kernel_size=1,
			padding='same',
			kernel_initializer=initializer,
			kernel_regularizer=regularizer,
			name='rpn_regression_head')
		self.reg_reshape = tf.keras.layers.Reshape(
			target_shape=(-1, 4),
			name='rpn_reshape_regression_head')

		# Losses
		self.crossentropy = tf.keras.losses.BinaryCrossentropy()
		self.huber = tf.keras.losses.Huber(
			delta=1.0,
			reduction=tf.keras.losses.Reduction.NONE)

		_, grid_height, grid_width, _ = self.feature_extractor.output_shape
		grid_shape = (grid_height, grid_width)
		self.anchors = generate_anchors(
			scales=scales,
			aspect_ratios=aspect_ratios,
			grid_shape=grid_shape,
			stride_shape=(16, 16), 
			base_anchor_shape=base_anchor_shape)
		self.image_shape = image_shape

	def call(self, image, training=False):
		'''
		Forward pass.

		Args:
			- image: The input image, i.e., a tensor of shape self.image_shape.
			- training: A boolean indicating whether the training version of the
				computation graph should be constructed.
		'''
		preprocessed_image = preprocess_input(image)
		preprocessed_image = tf.expand_dims(preprocessed_image, 0)
		feature_maps = self.feature_extractor(preprocessed_image, training=training)
		
		features = self.intermediate_layer(feature_maps)
		
		cls_output = self.cls_layer(features)
		cls_output = self.cls_reshape(cls_output)
		cls_output = tf.squeeze(cls_output, [0])

		reg_output = self.reg_layer(features)
		reg_output = self.reg_reshape(reg_output)
		reg_output = tf.squeeze(reg_output, [0])
		
		return cls_output, reg_output

	def postprocess_output(self, cls_output, reg_output, training):
		'''
		Postprocess the output of the RPN

		Args:
			- cls_output: The classification output of the RPN, i.e.,
				A Tensor of shape [num_anchors].
			- reg_output: The regression output of the RPN, i.e.,
				A Tensor of shape [num_anchors, 4].
			- training: A boolean value.

		Returns:
			roi_scores: A set of objectness scores for the rois.
			rois: A set of regions of interest, i.e., A set of box proposals.
		'''
		boxes = decode(reg_output, self.anchors)

		# Clip boxes to image boundaries
		image_height, image_width, _ = self.image_shape
		image_max_boudaries = tf.constant([image_width, image_height, image_width, image_height], dtype=tf.float32)

		boxes = tf.maximum(boxes, 0.)
		boxes = tf.minimum(boxes, image_max_boudaries)

		# Non Max Suppression
		inds_to_keep = tf.image.non_max_suppression(
		    boxes=boxes,
		    scores=cls_output,
		    max_output_size=2000 if training else 300,
		    iou_threshold=0.7)

		rois = tf.gather(boxes, inds_to_keep)
		roi_scores = tf.gather(cls_output, inds_to_keep)

		return roi_scores, rois

	def _classification_loss(self, target_labels, pred_scores):
		'''
		Computes the classification loss for a single image.

		Args:
			- target_labels: A tensor of shape [num_anchors].
			- pred_scores: A tensor of shape [num_anchors].
		'''
		inds_to_keep = tf.where(target_labels != -1)
		labels = tf.gather(target_labels, inds_to_keep)
		scores = tf.gather(pred_scores, inds_to_keep)
		
		return self.crossentropy(labels, scores)

	def _regression_loss(self, target_labels, target_boxes, pred_boxes):
		'''
		Computes the regression loss for a single image.

		Args:
			- target_labels: A tensor of shape [num_anchors].
			- target_boxes: A tensor of shape [num_anchors, 4].
			- pred_boxes: A tensor of shape [num_anchors, 4].
		'''
		reg_loss = self.huber(target_boxes, pred_boxes)
		reg_loss = tf.reduce_sum(reg_loss, -1)
		reg_loss = tf.gather(reg_loss, tf.where(target_labels == 1))
		reg_loss = tf.reduce_mean(reg_loss)

		return reg_loss

	@tf.function
	def train_step(self, image, boxes, optimizer, batch_size=256):
		target_boxes, target_labels = generate_targets(
			gt_boxes=boxes,
			anchor_boxes=self.anchors,
			image_shape=self.image_shape,
			num_anchors_to_keep=batch_size)
		
		with tf.GradientTape() as tape:
			cls_output, reg_output = self.call(image, training=True)
			
			cls_loss = self._classification_loss(target_labels, cls_output)
			reg_loss = self._regression_loss(target_labels, target_boxes, reg_output)
			
			multi_task_loss = cls_loss + reg_loss + sum(self.losses)

		gradients = tape.gradient(multi_task_loss, self.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return cls_loss, reg_loss

	@tf.function
	def test_step(self, image, boxes, batch_size=256):
		target_boxes, target_labels = generate_targets(
			gt_boxes=boxes,
			anchor_boxes=self.anchors,
			image_shape=self.image_shape,
			num_anchors_to_keep=batch_size)

		cls_output, reg_output = self.call(image)
			
		cls_loss = self._classification_loss(target_labels, cls_output)
		reg_loss = self._regression_loss(target_labels, target_boxes, reg_output)

		return cls_loss, reg_loss
