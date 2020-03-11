import tensorflow as tf 

class ClassificationLoss(tf.keras.losses.Loss):
	def __init__(self, name='classification_loss'):
		super(ClassificationLoss, self).__init__(name=name)

		self._cce = tf.keras.losses.CategoricalCrossentropy()

	def call(self, target_class_labels, pred_class_scores):
		'''
		Args:
			- target_class_labels: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1] 
				representing the target labels.
			- pred_class_scores: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1] 
				representing classification scores.
		'''
		return self._cce(target_class_labels, pred_class_scores)

class RegressionLoss(tf.keras.losses.Loss):
	def __init__(self, name='regression_loss'):
		super(RegressionLoss, self).__init__(name=name)

		self._huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

	def call(self, target_class_labels, target_boxes_encoded, pred_boxes_encoded):
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
