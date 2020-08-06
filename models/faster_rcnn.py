import tensorflow as tf

from models.abstract_detection_model import AbstractDetectionModel
from models.detectors.fast_rcnn_detector import FastRCNNDetector
from models.detectors.rpn_detector import RPNDetector
from models.feature_extractor import get_feature_extractor_model


class FasterRCNN(AbstractDetectionModel):
    def __init__(
        self,
        image_shape,
        num_classes,
        rpn_window_size=3,
        rpn_scales=[0.25, 0.5, 1.0, 2.0],
        rpn_aspect_ratios=[0.5, 1.0, 2.0],
        rpn_base_anchor_shape=(256, 256),
        name="faster_rcnn",
    ):
        """
        Instantiate a Faster-RCNN model.

        Args:
            - image_shape: Shape of the input images.
            - num_classes: Number of classes without background.
        """
        super(FasterRCNN, self).__init__(image_shape=image_shape, name=name)

        self._image_shape = image_shape

        self.feature_extractor = get_feature_extractor_model(image_shape)

        _, grid_height, grid_width, _ = self.feature_extractor.output_shape
        grid_shape = (grid_height, grid_width)

        self.rpn_detector = RPNDetector(
            image_shape=image_shape,
            grid_shape=grid_shape,
            window_size=rpn_window_size,
            scales=rpn_scales,
            aspect_ratios=rpn_aspect_ratios,
            base_anchor_shape=rpn_base_anchor_shape,
        )

        self.fast_rcnn_detector = FastRCNNDetector(image_shape, num_classes)

    def call(self, images, training=False):
        feature_maps = self.feature_extractor(images, training=training)

        rpn_output = self.rpn_detector(feature_maps, training=training)

        rois, _ = self.rpn_detector.postprocess_output(rpn_output, training=training)
        rcnn_output = self.fast_rcnn_detector(feature_maps, rois)

        return rpn_output, rcnn_output

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
        return self.fast_rcnn_detector.postprocess_output(rois, pred_class_scores, pred_boxes_encoded, training)

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
            - pred_scores: predicted class scores.
            - pred_classes: predicted class indices.
        """
        with tf.GradientTape() as tape:
            rpn_output, rcnn_output = self(images, True)

            # Get Fast RCNN training samples
            rpn_training_samples = self.rpn_detector.get_training_samples(gt_class_labels, gt_boxes, rpn_output)

            # Get Fast RCNN training samples
            rcnn_training_samples = self.fast_rcnn_detector.get_training_samples(gt_class_labels, gt_boxes, rcnn_output)

            # Get losses
            losses = {}
            losses["rpn_cls"] = self._classification_loss(
                rpn_training_samples["target_labels"], rpn_training_samples["pred_scores"]
            )
            losses["rpn_reg"] = self._regression_loss(
                rpn_training_samples["target_boxes"], rpn_training_samples["pred_boxes"]
            )
            losses["rcnn_cls"] = self._classification_loss(
                rcnn_training_samples["target_labels"], rcnn_training_samples["pred_scores"]
            )
            losses["rcnn_reg"] = self._regression_loss(
                rcnn_training_samples["target_boxes"], rcnn_training_samples["pred_boxes"]
            )
            multi_task_loss = sum(losses.values()) + sum(self.losses)

        gradients = tape.gradient(multi_task_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        rpn_pred_boxes, rpn_pred_scores = self.rpn_detector.postprocess_output(rpn_output, training=True)
        rcnn_pred_boxes, rcnn_pred_scores, rcnn_pred_classes = self.fast_rcnn_detector.postprocess_output(
            rcnn_output, training=True
        )

        preds = {
            "rpn_boxes": rpn_pred_boxes,
            "rpn_scores": rpn_pred_scores,
            "rcnn_boxes": rcnn_pred_boxes,
            "rcnn_scores": rcnn_pred_scores,
            "rcnn_classes": rcnn_pred_classes,
        }

        return losses, preds

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
            - pred_scores: predicted class scores.
            - pred_classes: predicted class indices.
            - foreground_rois: RoIs labeled as foreground.
            - background_rois: RoIs labeled as background.
        """
        rpn_output, rcnn_output = self(images, False)

        # Get Fast RCNN training samples
        rpn_training_samples = self.rpn_detector.get_training_samples(gt_class_labels, gt_boxes, rpn_output)

        # Get Fast RCNN training samples
        rcnn_training_samples = self.fast_rcnn_detector.get_training_samples(gt_class_labels, gt_boxes, rcnn_output)

        # Get losses
        losses = {}
        losses["rpn_cls"] = self._classification_loss(
            rpn_training_samples["target_labels"], rpn_training_samples["pred_scores"]
        )
        losses["rpn_reg"] = self._regression_loss(
            rpn_training_samples["target_boxes"], rpn_training_samples["pred_boxes"]
        )
        losses["rcnn_cls"] = self._classification_loss(
            rcnn_training_samples["target_labels"], rcnn_training_samples["pred_scores"]
        )
        losses["rcnn_reg"] = self._regression_loss(
            rcnn_training_samples["target_boxes"], rcnn_training_samples["pred_boxes"]
        )

        rpn_pred_boxes, rpn_pred_scores = self.rpn_detector.postprocess_output(rpn_output, training=False)
        rcnn_pred_boxes, rcnn_pred_scores, rcnn_pred_classes = self.fast_rcnn_detector.postprocess_output(
            rcnn_output, training=False
        )

        preds = {
            "rpn_boxes": rpn_pred_boxes,
            "rpn_scores": rpn_pred_scores,
            "rcnn_boxes": rcnn_pred_boxes,
            "rcnn_scores": rcnn_pred_scores,
            "rcnn_classes": rcnn_pred_classes,
        }

        return losses, preds
