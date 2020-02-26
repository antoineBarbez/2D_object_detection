import tensorflow as tf

from abc import abstractmethod
from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from models.target_generator import TargetGenerator 

class ObjectDetectionModel(tf.keras.Model):
	'''Abstract base class for object detection'''
	def __init__(self,
		image_shape, 
		num_classes,
		foreground_proportion,
		foreground_iou_interval, 
		background_iou_interval,  
		**kwargs):
		'''
		Instantiate an ObjectDetectionModel.

		Args:
			- image_shape: Shape of the input images.
			- num_classes: Number of classes without background.
			- foreground_proportion: Maximum proportion of foreground vs background  
				examples to sample in each minibatch. This parameter is set to 0.5 for
				the RPN and 0.25 for the Faster-RCNN according to the respective papers. 
			- foreground_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as foreground. 
			- background_iou_interval: Regions that have an IoU overlap with a ground-truth 
				bounding box in this interval are labeled as background.

			Note: Regions whose IoU overlap with a ground-truth bounding box is neither in
				foreground_iou_interval nor in background_iou_interval are ignored, i.e.,
				they do not contribute to the training objective.
		'''
		super(ObjectDetectionModel, self).__init__(**kwargs)
		self.image_shape = image_shape
		self.foreground_proportion = foreground_proportion

		self.target_generator = TargetGenerator(
			image_shape=image_shape,
			num_classes=num_classes,
			foreground_iou_interval=foreground_iou_interval,
			background_iou_interval=background_iou_interval)

		self.cce = tf.keras.losses.CategoricalCrossentropy()
		self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

		self.feature_extractor = ResNet50FeatureExtractor(
			kernel_regularizer=tf.keras.regularizers.l2(0.00005),
			input_shape=image_shape)
	
	def call(self, image, training=False, **kwargs):
		'''
		Forward pass.
		'''
		preprocessed_image = tf.expand_dims(image, 0)
		preprocessed_image = preprocess_input(preprocessed_image)

		feature_maps = self.feature_extractor(preprocessed_image, training=training)

		regions, pred_class_scores, pred_boxes_encoded = self._predict(feature_maps, training, **kwargs)
		
		return regions, pred_class_scores, pred_boxes_encoded
	
	@abstractmethod
	def postprocess_output(self, regions, pred_class_scores, pred_boxes_encoded, training):
		'''
		Postprocess the output of the object detection model.

		Args:
			- regions: Reference boxes to be used for decoding the output of the regression head.
				These correspond to the anchor boxes for the RPN model and to the RoI's for the
				Fast-RCNN model. shape = [num_regions, 4].
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[num_regions, num_classes, 4] representing encoded predicted box coordinates.
			- training: A boolean value indicating whether we are in training mode.

		Returns:
			pred_class_scores: A tensor of shape [num_boxes, num_classes + 1] representing
				the predicted class scores for each bounding box.
			pred_boxes: A tensor of shape [num_boxes, 4] representing the predicted bounding box
				coordinates.
		'''
		raise NotImplementedError

	@tf.function
	def train_step(self, image, gt_class_labels, gt_boxes, optimizer, 
		minibatch_size=256, **kwargs):
		'''
		Args:
			- image: Input image. A tensor of shape [height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [max_num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape [max_num_objects, 4].
			- optimizer: A tf.keras.optimizers.Optimizer object.
			- minibatch_size: Number of examples (regions) to sample per image.
			- **kwargs: Additional keyword arguments to be passed to the model's call() function.

		Returns:
			Two scalars, the training classification and regression losses. 
		'''
		with tf.GradientTape() as tape:
			regions, pred_class_scores, pred_boxes_encoded = self.call(image, True, **kwargs)
			
			target_class_labels, target_boxes_encoded = (
				self.target_generator.generate_targets(
					gt_class_labels=gt_class_labels,
					gt_boxes=gt_boxes,
					regions=regions))
			
			(target_class_labels, target_boxes_encoded, 
			 pred_class_scores, pred_boxes_encoded) = self._sample_minibatch(
				target_class_labels=target_class_labels,
				target_boxes_encoded=target_boxes_encoded,
				pred_class_scores=pred_class_scores,
				pred_boxes_encoded=pred_boxes_encoded,
				minibatch_size=minibatch_size,
				foreground_proportion=self.foreground_proportion)
			
			cls_loss = self.classification_loss(target_class_labels, pred_class_scores)
			reg_loss = self.regression_loss(
				target_class_labels, 
				target_boxes_encoded, 
				pred_boxes_encoded)
			multi_task_loss = cls_loss + reg_loss + sum(self.losses)

		gradients = tape.gradient(multi_task_loss, self.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return cls_loss, reg_loss

	@tf.function
	def test_step(self, image, gt_class_labels, gt_boxes, 
		minibatch_size=256, **kwargs):
		'''
		Args:
			- image: Input image. A tensor of shape [height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [max_num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape [max_num_objects, 4].
			- minibatch_size: Number of examples (regions) to sample per image.
			- **kwargs: Additional keyword arguments to be passed to the model's call() function.

		Returns:
			Two scalars, the test classification and regression losses. 
		'''
		regions, pred_class_scores, pred_boxes_encoded = self.call(image, False, **kwargs)
		
		target_class_labels, target_boxes_encoded = (
			self.target_generator.generate_targets(
				gt_class_labels=gt_class_labels,
				gt_boxes=gt_boxes,
				regions=regions))
		
		(target_class_labels, target_boxes_encoded, 
		 pred_class_scores, pred_boxes_encoded) = self._sample_minibatch(
			target_class_labels=target_class_labels,
			target_boxes_encoded=target_boxes_encoded,
			pred_class_scores=pred_class_scores,
			pred_boxes_encoded=pred_boxes_encoded,
			minibatch_size=minibatch_size,
			foreground_proportion=self.foreground_proportion) 
		
		cls_loss = self.classification_loss(target_class_labels, pred_class_scores)
		reg_loss = self.regression_loss(
			target_class_labels, 
			target_boxes_encoded, 
			pred_boxes_encoded)
		
		return cls_loss, reg_loss

	def classification_loss(self, target_class_labels, pred_class_scores):
		'''
		Args:
			- target_class_labels: A tensor of shape [minibatch_size, num_classes + 1] 
				representing the target labels.
			- pred_class_scores: A tensor of shape [minibatch_size, num_classes + 1] 
				representing classification scores.
		'''
		return self.cce(target_class_labels, pred_class_scores)

	def regression_loss(self, target_class_labels, target_boxes_encoded, pred_boxes_encoded):
		'''
		Args:
			- target_class_labels: A tensor of shape [minibatch_size, num_classes + 1] 
				representing the target labels.
			- target_boxes_encoded: A tensor of shape [minibatch_size, 4] representing the 
				encoded target ground-truth bounding boxes.
			- pred_boxes_encoded: A tensor of shape [minibatch_size, num_classes, 4] 
				representing the encoded predicted bounding boxess.
		'''
		foreground_inds = tf.reshape(tf.where(target_class_labels[:, 0] == 0.0), [-1])
		target_class_labels = tf.gather(target_class_labels, foreground_inds)
		target_boxes_encoded = tf.gather(target_boxes_encoded, foreground_inds)
		pred_boxes_encoded = tf.gather(pred_boxes_encoded, foreground_inds) 

		target_class_labels = target_class_labels[..., 1:]
		pred_boxes_encoded = tf.gather_nd(pred_boxes_encoded, tf.where(target_class_labels == 1.0))

		loss = self.huber(target_boxes_encoded, pred_boxes_encoded)
		loss = tf.reduce_sum(loss, -1)

		return tf.reduce_mean(loss)

	@abstractmethod
	def _predict(self, feature_maps, training, **kwargs):
		'''
		Args:
			- feature_maps: Output of the feature extractor.
			- training: A boolean indicating whether the training version of the
				computation graph should be constructed.

		Returns:
			- regions: Reference boxes to be used for decoding the output of the regression head.
				These correspond to the anchor boxes for the RPN model and to the RoI's for the
				Fast-RCNN model. shape = [num_regions, 4].
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[num_regions, num_classes + 1, 4] representing encoded predicted box coordinates.
		'''
		raise NotImplementedError

	def _sample_minibatch(self,
		target_class_labels,
		target_boxes_encoded,
		pred_class_scores,
		pred_boxes_encoded,
		minibatch_size,
		foreground_proportion):
		''' 
		Args:
			- target_class_labels: A tensor of shape [num_regions, num_classes + 1] representing the target
				labels for each region. The target label for ignored regions is zeros(num_classes + 1).
			- target_boxes_encoded: A tensor of shape [num_regions, 4] representing the encoded target ground-truth 
				bounding box for each region, i.e., the ground-truth bounding box with the highest IoU 
				overlap with the considered region. 
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[num_regions, num_classes + 1, 4] representing encoded predicted box coordinates.
			- minibatch_size: Number of examples (regions) to sample per image.
			- foreground_proportion: Maximum proportion of foreground vs background  
				examples to sample in each minibatch. This parameter is set to 0.5 for
				the RPN and 0.25 for the Faster-RCNN according to the respective papers.

		Returns:
			sampled target_class_labels, target_boxes_encoded, pred_class_scores, and 
			pred_boxes_encoded. The first dimension of these tensors equals minibatch_size.
		'''

		foreground_inds = tf.reshape(tf.where(
			(tf.reduce_sum(target_class_labels, -1) != 0.0) &
			(target_class_labels[:, 0] == 0.0)), [-1])

		background_inds = tf.reshape(tf.where(
			(tf.reduce_sum(target_class_labels, -1) != 0.0) &
			(target_class_labels[:, 0] == 1.0)), [-1])

		foreground_inds = tf.random.shuffle(foreground_inds)
		background_inds = tf.random.shuffle(background_inds)

		num_foreground_regions = tf.size(foreground_inds)
		num_background_regions = tf.size(background_inds)

		num_foreground_regions_to_keep = tf.minimum(
			num_foreground_regions,
			tf.cast(tf.math.round(minibatch_size*foreground_proportion), dtype=tf.int32))
		num_background_regions_to_keep = minibatch_size - num_foreground_regions_to_keep
		
		inds_to_keep = tf.concat([
			foreground_inds[:num_foreground_regions_to_keep],
			background_inds[:num_background_regions_to_keep]], 0)
		inds_to_keep.set_shape([minibatch_size])

		target_class_labels_sample = tf.gather(target_class_labels, inds_to_keep)
		pred_class_scores_sample = tf.gather(pred_class_scores, inds_to_keep)
		target_boxes_encoded_sample = tf.gather(target_boxes_encoded, inds_to_keep)
		pred_boxes_encoded_sample = tf.gather(pred_boxes_encoded, inds_to_keep)
		

		return (target_class_labels_sample, target_boxes_encoded_sample, 
			pred_class_scores_sample, pred_boxes_encoded_sample) 
