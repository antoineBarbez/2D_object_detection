import tensorflow as tf


class ClassificationLoss(tf.keras.losses.Loss):
    def __init__(self, name="classification_loss"):
        super(ClassificationLoss, self).__init__(name=name)

        self._cce = tf.keras.losses.CategoricalCrossentropy()

    def __call__(self, target_class_labels, pred_class_scores):
        """
        Args:
            - target_class_labels: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1]
                representing the target labels.
            - pred_class_scores: A tensor of shape [batch_size, num_samples_per_image, num_classes + 1]
                representing classification scores.
        """
        return self._cce(target_class_labels, pred_class_scores)


class RegressionLoss(tf.keras.losses.Loss):
    def __init__(self, name="regression_loss"):
        super(RegressionLoss, self).__init__(name=name)

        self._huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self, target_boxes_encoded, pred_boxes_encoded):
        """
        Args:
            - target_boxes_encoded: A tensor of shape [batch_size, num_samples_per_image, num_classes, 4]
                representing the encoded target ground-truth bounding boxes.
            - pred_boxes_encoded: A tensor of shape [batch_size, num_samples_per_image, num_classes, 4]
                representing the encoded predicted bounding boxess.
        """
        inds_to_keep = tf.where(tf.reduce_sum(target_boxes_encoded, axis=-1) != 0.0)
        target_boxes_encoded = tf.gather_nd(target_boxes_encoded, inds_to_keep)
        pred_boxes_encoded = tf.gather_nd(pred_boxes_encoded, inds_to_keep)

        loss = self._huber(target_boxes_encoded, pred_boxes_encoded)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)

        return loss
