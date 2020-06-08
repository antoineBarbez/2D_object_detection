import tensorflow as tf

from models.abstract_detection_model import AbstractDetectionModel
from models.detectors.rpn_detector import RPNDetector
from models.feature_extractor import get_feature_extractor_model


class RPN(AbstractDetectionModel):
    def __init__(
        self,
        image_shape,
        window_size=3,
        scales=[0.25, 0.5, 1.0, 2.0],
        aspect_ratios=[0.5, 1.0, 2.0],
        base_anchor_shape=(256, 256),
        name="region_proposal_network",
    ):
        """
        Instantiate a Region Proposal Network.

        Args:
            - image_shape: Shape of the input images.
            - window_size: Size of the sliding window.
            - scales: Anchors' scales.
            - aspect_ratios: Anchors' aspect ratios.
            - base_anchor_shape: Shape of the base anchor.
        """
        super(RPN, self).__init__(image_shape=image_shape, name=name)

        self.feature_extractor = get_feature_extractor_model(image_shape)

        _, grid_height, grid_width, _ = self.feature_extractor.output_shape
        grid_shape = (grid_height, grid_width)

        self.detector = RPNDetector(
            image_shape=image_shape,
            grid_shape=grid_shape,
            window_size=window_size,
            scales=scales,
            aspect_ratios=aspect_ratios,
            base_anchor_shape=base_anchor_shape,
        )

    def call(self, images, training):
        # preprocessed_images = preprocess_input(images)

        feature_maps = self.feature_extractor(images, training=training)

        return self.detector(feature_maps, training=training)

    def postprocess_output(self, anchors, pred_class_scores, pred_boxes_encoded, training=False):
        """
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
        """
        return self.detector.postprocess_output(anchors, pred_class_scores, pred_boxes_encoded, training)

    @tf.function
    def train_step(self, images, gt_class_labels, gt_boxes, optimizer):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_class_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - optimizer: A tf.keras.optimizers.Optimizer object.
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            - cls_loss: A scalar, the training classification loss.
            - reg_loss: A scalar, the training regression loss.
            - pred_boxes: predicted box proposals.
            - pred_scores: predicted objectness scores.
        """
        with tf.GradientTape() as tape:
            anchors, pred_objectness_scores, pred_boxes_encoded = self(images, True)

            (
                target_objectness_labels_sample,
                target_boxes_encoded_sample,
                pred_objectness_scores_sample,
                pred_boxes_encoded_sample,
                _,
            ) = self.detector.get_training_samples(
                anchors, gt_class_labels, gt_boxes, pred_objectness_scores, pred_boxes_encoded
            )

            cls_loss = self._classification_loss(target_objectness_labels_sample, pred_objectness_scores_sample)
            reg_loss = self._regression_loss(
                target_objectness_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
            )
            multi_task_loss = cls_loss + reg_loss + sum(self.losses)

        gradients = tape.gradient(multi_task_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        pred_boxes, pred_scores = self.postprocess_output(anchors, pred_objectness_scores, pred_boxes_encoded, False)

        return cls_loss, reg_loss, pred_boxes, pred_scores

    @tf.function
    def test_step(self, images, gt_class_labels, gt_boxes):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_class_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            - cls_loss: A scalar, the training classification loss.
            - reg_loss: A scalar, the training regression loss.
            - pred_boxes: predicted box proposals.
            - pred_scores: predicted objectness scores.
            - foreground_anchors: Anchors labeled as foreground.
            - background_anchors: Anchors labeled as background.
        """
        anchors, pred_objectness_scores, pred_boxes_encoded = self(images, False)

        (
            target_objectness_labels_sample,
            target_boxes_encoded_sample,
            pred_objectness_scores_sample,
            pred_boxes_encoded_sample,
            anchors_sample,
        ) = self.detector.get_training_samples(
            anchors, gt_class_labels, gt_boxes, pred_objectness_scores, pred_boxes_encoded
        )

        cls_loss = self._classification_loss(target_objectness_labels_sample, pred_objectness_scores_sample)
        reg_loss = self._regression_loss(
            target_objectness_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
        )

        pred_boxes, pred_scores = self.postprocess_output(anchors, pred_objectness_scores, pred_boxes_encoded, False)

        foreground_inds = tf.where(target_objectness_labels_sample[..., 0] == 0.0)
        foreground_anchors = tf.gather_nd(anchors_sample, foreground_inds)

        background_inds = tf.where(target_objectness_labels_sample[..., 0] == 1.0)
        background_anchors = tf.gather_nd(anchors_sample, background_inds)

        return cls_loss, reg_loss, pred_boxes, pred_scores, foreground_anchors, background_anchors
