import tensorflow as tf

from abc import abstractmethod
from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from utils.targets import TargetGenerator 

class ObjectDetectionModel(tf.keras.Model):
	'''Abstract base class for object detection'''
	def __init__(self,
		image_shape, 
		num_classes,
		foreground_proportion,
		foreground_iou_interval, 
		background_iou_interval,
		log_dir=None,
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
		self._image_shape = image_shape
		self._foreground_proportion = foreground_proportion
		self._target_generator = TargetGenerator(
			image_shape=image_shape,
			num_classes=num_classes,
			foreground_iou_interval=foreground_iou_interval,
			background_iou_interval=background_iou_interval)

		self._feature_extractor = ResNet50FeatureExtractor(
			kernel_regularizer=tf.keras.regularizers.l2(0.00005),
			input_shape=image_shape)

		self._detector = NotImplemented

		self._cce = tf.keras.losses.CategoricalCrossentropy()
		self._huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)


		'''self.train_classification_loss = tf.keras.metrics.Mean(name='train_classification_loss')
		self.train_regression_loss = tf.keras.metrics.Mean(name='train_regression_loss')
		self.train_average_precision = metric_utils.AveragePrecision(0.5, name='train_ap_0.5')

		self.valid_classification_loss = tf.keras.metrics.Mean(name='valid_classification_loss')
		self.valid_regression_loss = tf.keras.metrics.Mean(name='valid_regression_loss')
		self.valid_average_precision = metric_utils.AveragePrecision(0.5, name='valid_ap_0.5')

		if log_dir is not None:
			train_log_dir = os.path.join(log_dir, self.name, 'train')
			valid_log_dir = os.path.join(log_dir, self.name, 'valid')

			self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
			self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)'''


	@property
	def detector(self):
		return self._detector
	
	def call(self, images, training=False, **kwargs):
		'''
		Forward pass.

		Args:
			- images: A batch of input images. A [batch_size, height, width, channels] tensor.
			- training: Boolean value indicating whether the training version of the computation
				graph should be constructed.
			- **kwargs: Additional parameters to be passed to the detector's call() method.

		Returns:
			- regions: A tensor of shape [num_regions, 4] representing the reference box coordinates
				to be used for decoding the predicted boxes.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_regions, num_classes, 4] representing encoded predicted box coordinates. 
		'''
		preprocessed_images = preprocess_input(images)

		feature_maps = self._feature_extractor(preprocessed_images, training=training)

		regions, pred_class_scores, pred_boxes_encoded = self._detector(feature_maps, training, **kwargs)
		
		return regions, pred_class_scores, pred_boxes_encoded

	def postprocess_output(self, regions, pred_class_scores, pred_boxes_encoded, **kwargs):
		'''
		Args:
			- regions: A tensor of shape [num_regions, 4] representing the reference box coordinates
				to be used for decoding the predicted boxes.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_regions, num_classes + 1] representing classification scores.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_regions, num_classes, 4] representing encoded predicted box coordinates.
		'''
		return self._detector.postprocess_output(regions, pred_class_scores, pred_boxes_encoded, **kwargs)

	@tf.function
	def predict(self, images, **kwargs):
		image_height, image_width, _ = self._image_shape
		resized_images = tf.image.resize(images, [image_height, image_width])
		regions, pred_class_scores, pred_boxes_encoded = self.call(resized_images, **kwargs)

		return self.postprocess_output(regions, pred_class_scores, pred_boxes_encoded)

	def classification_loss(self, target_class_labels, pred_class_scores):
		'''
		Args:
			- target_class_labels: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1] 
				representing the target labels.
			- pred_class_scores: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1] 
				representing classification scores.
		'''
		return self._cce(target_class_labels, pred_class_scores)

	def regression_loss(self, target_class_labels, target_boxes_encoded, pred_boxes_encoded):
		'''
		Args:
			- target_class_labels: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1] 
				representing the target labels.
			- target_boxes_encoded: A tensor of shape [batch_size, num_samples_per_image, 4]
				representing the encoded target ground-truth bounding boxes.
			- pred_boxes_encoded: A tensor of shape [batch_size, num_samples_per_image, num_classes, 4] 
				representing the encoded predicted bounding boxess.
		'''
		foreground_inds = tf.where(target_class_labels[..., 0] == 0.0)
		target_class_labels = tf.gather_nd(target_class_labels, foreground_inds)
		target_boxes_encoded = tf.gather_nd(target_boxes_encoded, foreground_inds)
		pred_boxes_encoded = tf.gather_nd(pred_boxes_encoded, foreground_inds) 

		target_class_labels = target_class_labels[..., 1:]
		pred_boxes_encoded = tf.gather_nd(pred_boxes_encoded, tf.where(target_class_labels == 1.0))

		loss = self._huber(target_boxes_encoded, pred_boxes_encoded)
		loss = tf.reduce_sum(loss, -1)

		return tf.reduce_mean(loss)

	@tf.function
	def train_step(self, images, gt_class_labels, gt_boxes, optimizer, 
		num_samples_per_image=256, **kwargs):
		'''
		Args:
			- images: Input images. A tensor of shape [batch_size, height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [batch_size, max_num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
				[batch_size, max_num_objects, 4].
			- optimizer: A tf.keras.optimizers.Optimizer object.
			- num_samples_per_image: Number of examples (regions) to sample per image.
			- **kwargs: Additional keyword arguments to be passed to the model's call() method.

		Returns:
			Two scalars, the training classification and regression losses. 
		'''
		with tf.GradientTape() as tape:
			regions, pred_class_scores, pred_boxes_encoded = self.call(images, True, **kwargs)

			target_class_labels, target_boxes_encoded = (
				self._target_generator.generate_targets_batch(
					gt_class_labels=gt_class_labels,
					gt_boxes=gt_boxes,
					regions=regions))
			
			(target_class_labels_sample, target_boxes_encoded_sample, 
			 pred_class_scores_sample, pred_boxes_encoded_sample, _) = self._sample_batch(
				target_class_labels=target_class_labels,
				target_boxes_encoded=target_boxes_encoded,
				pred_class_scores=pred_class_scores,
				pred_boxes_encoded=pred_boxes_encoded,
				regions=regions,
				num_samples_per_image=num_samples_per_image,
				foreground_proportion=self._foreground_proportion)
			
			cls_loss = self.classification_loss(target_class_labels_sample, pred_class_scores_sample)
			reg_loss = self.regression_loss(
				target_class_labels_sample, 
				target_boxes_encoded_sample, 
				pred_boxes_encoded_sample)
			multi_task_loss = cls_loss + reg_loss + sum(self.losses)

		gradients = tape.gradient(multi_task_loss, self.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		pred_scores, pred_boxes = self.postprocess_output(regions, pred_class_scores, pred_boxes_encoded)

		return cls_loss, reg_loss, pred_scores, pred_boxes

	@tf.function
	def test_step(self, images, gt_class_labels, gt_boxes, 
		num_samples_per_image=256, **kwargs):
		'''
		Args:
			- images: Input images. A tensor of shape [batch_size, height, width, 3].
			- gt_class_labels: Ground-truth class labels padded with background one hot encoded.
				A tensor of shape [batch_size, max_num_objects, num_classes + 1].
			- gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
				[batch_size, max_num_objects, 4].
			- num_samples_per_image: Number of examples (regions) to sample per image.
			- **kwargs: Additional keyword arguments to be passed to the model's call() method.

		Returns:
			Two scalars, the test classification and regression losses. 
		'''
		regions, pred_class_scores, pred_boxes_encoded = self.call(images, False, **kwargs)

		target_class_labels, target_boxes_encoded = (
			self._target_generator.generate_targets_batch(
				gt_class_labels=gt_class_labels,
				gt_boxes=gt_boxes,
				regions=regions))
		
		(target_class_labels_sample, target_boxes_encoded_sample, 
		 pred_class_scores_sample, pred_boxes_encoded_sample, regions_sample) = self._sample_batch(
			target_class_labels=target_class_labels,
			target_boxes_encoded=target_boxes_encoded,
			pred_class_scores=pred_class_scores,
			pred_boxes_encoded=pred_boxes_encoded,
			regions=regions,
			num_samples_per_image=num_samples_per_image,
			foreground_proportion=self._foreground_proportion) 
		
		cls_loss = self.classification_loss(target_class_labels_sample, pred_class_scores_sample)
		reg_loss = self.regression_loss(
			target_class_labels_sample, 
			target_boxes_encoded_sample, 
			pred_boxes_encoded_sample)

		pred_scores, pred_boxes = self.postprocess_output(regions, pred_class_scores, pred_boxes_encoded)

		foreground_inds = tf.where(target_class_labels_sample[..., 0] == 0.0)
		foreground_regions = tf.gather_nd(regions_sample, foreground_inds)

		background_inds = tf.where(target_class_labels_sample[..., 0] == 1.0)
		background_regions = tf.gather_nd(regions_sample, background_inds)

		return cls_loss, reg_loss, pred_scores, pred_boxes, foreground_regions, background_regions

	def _sample_batch(self,
		target_class_labels,
		target_boxes_encoded,
		pred_class_scores,
		pred_boxes_encoded,
		regions,
		num_samples_per_image,
		foreground_proportion):

		return tf.map_fn(
		 	fn=lambda x: self._sample_image(x[0], x[1], x[2], x[3], regions, num_samples_per_image, foreground_proportion),
		 	elems=(target_class_labels, target_boxes_encoded, pred_class_scores, pred_boxes_encoded),
		 	dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

	def _sample_image(self,
		target_class_labels,
		target_boxes_encoded,
		pred_class_scores,
		pred_boxes_encoded,
		regions,
		num_samples,
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
			- num_samples: Number of examples (regions) to sample per image.
			- foreground_proportion: Maximum proportion of foreground vs background  
				examples to sample in each minibatch. This parameter is set to 0.5 for
				the RPN and 0.25 for the Faster-RCNN according to the respective papers.

		Returns:
			sampled target_class_labels, target_boxes_encoded, pred_class_scores, and 
			pred_boxes_encoded. The first dimension of these tensors equals num_samples.
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
			tf.cast(tf.math.round(num_samples*foreground_proportion), dtype=tf.int32))
		num_background_regions_to_keep = num_samples - num_foreground_regions_to_keep
		
		foreground_inds = foreground_inds[:num_foreground_regions_to_keep]
		background_inds = background_inds[:num_background_regions_to_keep]

		inds_to_keep = tf.concat([foreground_inds, background_inds], 0)
		inds_to_keep.set_shape([num_samples])

		target_class_labels_sample = tf.gather(target_class_labels, inds_to_keep)
		pred_class_scores_sample = tf.gather(pred_class_scores, inds_to_keep)
		target_boxes_encoded_sample = tf.gather(target_boxes_encoded, inds_to_keep)
		pred_boxes_encoded_sample = tf.gather(pred_boxes_encoded, inds_to_keep)
		regions_sample = tf.gather(regions, inds_to_keep)

		return (target_class_labels_sample, target_boxes_encoded_sample, 
			pred_class_scores_sample, pred_boxes_encoded_sample, regions_sample)
