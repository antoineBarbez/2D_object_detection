import tensorflow as tf
import utils.sampling as sampling_utils

from models.abstract_detection_model import AbstractDetectionModel
from models.detectors.fast_rcnn_detector import FastRCNNDetector
from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from utils.targets import TargetGenerator

import utils.boxes as box_utils


class FastRCNN(AbstractDetectionModel):
    def __init__(self, image_shape, num_classes, name="fast_rcnn"):
        """
        Instantiate a Fast-RCNN model.

        Args:
            - image_shape: Shape of the input images.
            - num_classes: Number of classes without background.
        """
        super(FastRCNN, self).__init__(image_shape=image_shape, name=name)

        self._target_generator = TargetGenerator(
            image_shape=image_shape,
            num_classes=num_classes,
            foreground_iou_interval=(0.5, 1.0),
            background_iou_interval=(0.0, 0.5),
        )

        self.feature_extractor = ResNet50FeatureExtractor(
            kernel_regularizer=tf.keras.regularizers.l2(0.00005), input_shape=image_shape
        )

        self.detector = FastRCNNDetector(image_shape, num_classes)

    def call(self, images, rois, training=False):
        preprocessed_images = preprocess_input(images)

        feature_maps = self.feature_extractor(preprocessed_images, training=training)

        return self.detector(feature_maps, rois)

    def postprocess_output(self, rois, pred_class_scores, pred_boxes_encoded, training=False):
        """
        Postprocess the output of the Fast-RCNN

        Args:
            - rois: A tensor of shape [num_rois, 4] representing the Regions of Interest
                in absolute coordinates.
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_rois, num_classes + 1] representing classification scores 
                for each Region of Interest.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape 
                [batch_size, num_rois, num_classes, 4] representing encoded predicted box 
                coordinates for each Region of Interest.

        Returns:
            - boxes: A [batch_size, max_total_size, 4] float32 tensor containing the 
                predicted bounding boxes.
            - scores: A [batch_size, max_total_size] float32 tensor containing the 
                class scores for the boxes.
            - classes: A [batch_size, max_total_size] float32 tensor containing the 
                class indices for the boxes.

        With max_total_size = 300.
        """
        return self.detector.postprocess_output(rois, pred_class_scores, pred_boxes_encoded, training)

    @tf.function
    def train_step(self, images, rois, gt_class_labels, gt_boxes, optimizer, num_samples_per_image):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - rois: Regions of Interest. A tensor of shape [batch_size, num_rois, 4].
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
            - pred_scores: predicted class scores.
            - pred_classes: predicted class indices.
        """
        with tf.GradientTape() as tape:
            rois, pred_class_scores, pred_boxes_encoded = self.call(images, rois, True)

            target_class_labels, target_boxes_encoded = tf.map_fn(
                fn=lambda x: self._target_generator.generate_targets(x[0], x[1], x[2]),
                elems=(gt_class_labels, gt_boxes, rois),
                dtype=(tf.float32, tf.float32),
            )

            (
                target_class_labels_sample,
                target_boxes_encoded_sample,
                pred_class_scores_sample,
                pred_boxes_encoded_sample,
                _,
            ) = tf.map_fn(
                fn=lambda x: sampling_utils.sample_image(x[0], x[1], x[2], x[3], x[4], num_samples_per_image, 0.25),
                elems=(target_class_labels, target_boxes_encoded, pred_class_scores, pred_boxes_encoded, rois),
                dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            )

            cls_loss = self._classification_loss(target_class_labels_sample, pred_class_scores_sample)
            reg_loss = self._regression_loss(
                target_class_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
            )
            multi_task_loss = cls_loss + reg_loss + sum(self.losses)

        gradients = tape.gradient(multi_task_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        pred_boxes, pred_scores, pred_classes = self.postprocess_output(rois, pred_class_scores, pred_boxes_encoded)

        return cls_loss, reg_loss, pred_boxes, pred_scores, pred_classes

    @tf.function
    def test_step(self, images, rois, gt_class_labels, gt_boxes, num_samples_per_image):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - rois: Regions of Interest. A tensor of shape [batch_size, num_rois, 4].
            - gt_class_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            - cls_loss: A scalar, the training classification loss.
            - reg_loss: A scalar, the training regression loss.
            - pred_boxes: predicted box proposals.
            - pred_scores: predicted class scores.
            - pred_classes: predicted class indices.
            - foreground_rois: RoIs labeled as foreground.
            - background_rois: RoIs labeled as background.
        """
        rois, pred_class_scores, pred_boxes_encoded = self.call(images, rois, False)

        target_class_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: self._target_generator.generate_targets(x[0], x[1], x[2]),
            elems=(gt_class_labels, gt_boxes, rois),
            dtype=(tf.float32, tf.float32),
        )

        (
            target_class_labels_sample,
            target_boxes_encoded_sample,
            pred_class_scores_sample,
            pred_boxes_encoded_sample,
            rois_sample,
        ) = tf.map_fn(
            fn=lambda x: sampling_utils.sample_image(x[0], x[1], x[2], x[3], x[4], num_samples_per_image, 0.25),
            elems=(target_class_labels, target_boxes_encoded, pred_class_scores, pred_boxes_encoded, rois),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        cls_loss = self._classification_loss(target_class_labels_sample, pred_class_scores_sample)
        reg_loss = self._regression_loss(
            target_class_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
        )

        # pred_boxes_encoded = tf.tile(tf.expand_dims(target_boxes_encoded, 2), [1, 1, 7, 1])
        pred_boxes, pred_scores, pred_classes = self.postprocess_output(rois, pred_class_scores, pred_boxes_encoded)

        foreground_inds = tf.where(target_class_labels_sample[..., 0] == 0.0)
        foreground_rois = tf.gather_nd(rois_sample, foreground_inds)

        background_inds = tf.where(target_class_labels_sample[..., 0] == 1.0)
        background_rois = tf.gather_nd(rois_sample, background_inds)

        return cls_loss, reg_loss, pred_boxes, pred_scores, pred_classes, foreground_rois, background_rois
