import tensorflow as tf

from abc import abstractmethod
from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from models.utils.target_generator import TargetGenerator 

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
		preprocessed_image = tf.expand_dims(image, 0)
		preprocessed_image = preprocess_input(preprocessed_image)

		feature_maps = self.feature_extractor(preprocessed_image, training=training)

		regions, pred_scores, pred_boxes = self._predict(feature_maps, training, **kwargs)
		
		return regions, pred_scores, pred_boxes
	
	@abstractmethod
	def postprocess_output(self, regions, pred_scores, pred_boxes, training):
		raise NotImplementedError
	
	@tf.function
	def train_step(self, image, gt_labels, gt_boxes, optimizer, minibatch_size=256, **kwargs):
		with tf.GradientTape() as tape:
			regions, pred_scores, pred_boxes = self.call(image, True, **kwargs)
			
			target_labels, target_boxes = self.target_generator.generate_targets(
				gt_labels=gt_labels,
				gt_boxes=gt_boxes,
				regions=regions)
			
			target_labels, target_boxes, pred_scores, pred_boxes = self._sample_minibatch(
				target_labels=target_labels,
				target_boxes=target_boxes,
				pred_scores=pred_scores,
				pred_boxes=pred_boxes,
				minibatch_size=minibatch_size) 
			
			cls_loss = self.classification_loss(target_labels, pred_scores)
			reg_loss = self.regression_loss(target_labels, target_boxes, pred_boxes)
			multi_task_loss = cls_loss + reg_loss + sum(self.losses)

		gradients = tape.gradient(multi_task_loss, self.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return cls_loss, reg_loss

	@tf.function
	def test_step(self, image, gt_labels, gt_boxes, minibatch_size=256, **kwargs):
		regions, pred_scores, pred_boxes = self.call(image, False, **kwargs)
		
		target_labels, target_boxes = self.target_generator.generate_targets(
			gt_labels=gt_labels,
			gt_boxes=gt_boxes,
			regions=regions)
		
		target_labels, target_boxes, pred_scores, pred_boxes = self._sample_minibatch(
			target_labels=target_labels,
			target_boxes=target_boxes,
			pred_scores=pred_scores,
			pred_boxes=pred_boxes,
			minibatch_size=minibatch_size)
			
		cls_loss = self.classification_loss(target_labels, pred_scores)
		reg_loss = self.regression_loss(target_labels, target_boxes, pred_boxes)
		
		return cls_loss, reg_loss

	def classification_loss(self, target_labels, pred_scores):
		return self.cce(target_labels, pred_scores)

	def regression_loss(self, target_labels, target_boxes, pred_boxes):
		foreground_inds = tf.reshape(tf.where(target_labels[:, 0] == 0.0), [-1])
		target_labels = tf.gather(target_labels, foreground_inds)
		target_boxes = tf.gather(target_boxes, foreground_inds)
		pred_boxes = tf.gather(pred_boxes, foreground_inds) 

		target_labels = target_labels[..., 1:]
		pred_boxes = tf.gather_nd(pred_boxes, tf.where(target_labels == 1.0))

		loss = self.huber(target_boxes, pred_boxes)
		loss = tf.reduce_sum(loss, -1)

		return tf.reduce_mean(loss)

	@abstractmethod
	def _predict(self, feature_maps, training, **kwargs):
		raise NotImplementedError

	def _sample_minibatch(self,
		target_labels,
		target_boxes,
		pred_scores,
		pred_boxes,
		minibatch_size):

		foreground_inds = tf.reshape(tf.where(
			(tf.reduce_sum(target_labels, -1) != 0.0) &
			(target_labels[:, 0] == 0.0)), [-1])

		background_inds = tf.reshape(tf.where(
			(tf.reduce_sum(target_labels, -1) != 0.0) &
			(target_labels[:, 0] == 1.0)), [-1])

		foreground_inds = tf.random.shuffle(foreground_inds)
		background_inds = tf.random.shuffle(background_inds)

		num_foreground_regions = tf.size(foreground_inds)
		num_background_regions = tf.size(background_inds)

		num_foreground_regions_to_keep = tf.minimum(
			num_foreground_regions,
			tf.cast(tf.math.round(minibatch_size*self.foreground_proportion), dtype=tf.int32))
		num_background_regions_to_keep = minibatch_size - num_foreground_regions_to_keep
		
		inds_to_keep = tf.concat([
			foreground_inds[:num_foreground_regions_to_keep],
			background_inds[:num_background_regions_to_keep]], 0)
		inds_to_keep.set_shape([minibatch_size])

		target_labels_sample = tf.gather(target_labels, inds_to_keep)
		pred_scores_sample = tf.gather(pred_scores, inds_to_keep)
		target_boxes_sample = tf.gather(target_boxes, inds_to_keep)
		pred_boxes_sample = tf.gather(pred_boxes, inds_to_keep)
		

		return (target_labels_sample, target_boxes_sample, 
			pred_scores_sample, pred_boxes_sample) 
