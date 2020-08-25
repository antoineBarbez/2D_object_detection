import tensorflow as tf

from models.detectors.fast_rcnn_detector import FastRCNNDetector
from models.detectors.rpn_detector import RPNDetector
from models.feature_extractor import get_feature_extractor_model
from utils.losses import ClassificationLoss, RegressionLoss
from utils.post_processing import postprocess_output


class FasterRCNN(tf.keras.Model):
    def __init__(
        self, config, name="faster_rcnn",
    ):
        """
        Instantiate a Faster-RCNN model.

        Args:
            - config: Faster-RCNN configuration dictionary.
        """
        super(FasterRCNN, self).__init__(name=name)
        self._image_shape = config["image_shape"]
        self._rpn_config = config["rpn"]
        self._rcnn_config = config["rcnn"]

        self.feature_extractor = get_feature_extractor_model(self._image_shape)

        self.rpn_detector = RPNDetector(
            image_shape=self._image_shape,
            feature_maps_shape=self.feature_extractor.output_shape,
            config=self._rpn_config,
        )
        self.rcnn_detector = FastRCNNDetector(
            image_shape=self._image_shape, num_classes=config["num_classes"], config=self._rcnn_config
        )

        self._classification_loss = ClassificationLoss()
        self._regression_loss = RegressionLoss()

    def call(self, images, training=False):
        feature_maps = self.feature_extractor(images, training=training)

        rpn_output = self.rpn_detector(feature_maps, training=training)
        nmsed_rpn_output = postprocess_output(self._image_shape, **rpn_output, **self._rpn_config["nms"])

        rcnn_output = self.rcnn_detector(feature_maps, nmsed_rpn_output["pred_boxes"])

        return rpn_output, rcnn_output

    @tf.function
    def train_step(self, images, gt_labels, gt_boxes, optimizer):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape
                [batch_size, num_objects, 4].
            - optimizer: A tf.keras.optimizers.Optimizer object.

        Returns:
            Training losses and predictions.
        """
        with tf.GradientTape() as tape:
            rpn_output, rcnn_output = self(images, True)

            # Get Fast RCNN training samples
            rpn_training_samples = self.rpn_detector.get_training_samples(
                gt_labels, gt_boxes, **rpn_output, **self._rpn_config["sampling"]
            )

            # Get Fast RCNN training samples
            rcnn_training_samples = self.rcnn_detector.get_training_samples(
                gt_labels, gt_boxes, **rcnn_output, **self._rcnn_config["sampling"]
            )

            # Get losses
            losses = {
                "rpn_cls": self._classification_loss(
                    rpn_training_samples["target_labels"], rpn_training_samples["pred_scores"]
                ),
                "rpn_reg": self._regression_loss(
                    rpn_training_samples["target_boxes"], rpn_training_samples["pred_boxes"]
                ),
                "rcnn_cls": self._classification_loss(
                    rcnn_training_samples["target_labels"], rcnn_training_samples["pred_scores"]
                ),
                "rcnn_reg": self._regression_loss(
                    rcnn_training_samples["target_boxes"], rcnn_training_samples["pred_boxes"]
                ),
            }
            multi_task_loss = sum(losses.values()) + sum(self.losses)

        gradients = tape.gradient(multi_task_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        nmsed_rpn_output = postprocess_output(self._image_shape, **rpn_output, **self._rpn_config["nms"])
        nmsed_rcnn_output = postprocess_output(self._image_shape, **rcnn_output, **self._rcnn_config["nms"])

        preds = {
            "rpn_boxes": nmsed_rpn_output["pred_boxes"],
            "rpn_scores": nmsed_rpn_output["pred_scores"],
            "rcnn_boxes": nmsed_rcnn_output["pred_boxes"],
            "rcnn_scores": nmsed_rcnn_output["pred_scores"],
            "rcnn_classes": nmsed_rcnn_output["pred_classes"],
        }

        return losses, preds

    @tf.function
    def test_step(self, images, gt_labels, gt_boxes):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            Test losses and predictions.
        """
        rpn_output, rcnn_output = self(images, False)

        # Get Fast RCNN training samples
        rpn_training_samples = self.rpn_detector.get_training_samples(
            gt_labels, gt_boxes, **rpn_output, **self._rpn_config["sampling"]
        )

        # Get Fast RCNN training samples
        rcnn_training_samples = self.rcnn_detector.get_training_samples(
            gt_labels, gt_boxes, **rcnn_output, **self._rcnn_config["sampling"]
        )

        # Get losses
        losses = {
            "rpn_cls": self._classification_loss(
                rpn_training_samples["target_labels"], rpn_training_samples["pred_scores"]
            ),
            "rpn_reg": self._regression_loss(rpn_training_samples["target_boxes"], rpn_training_samples["pred_boxes"]),
            "rcnn_cls": self._classification_loss(
                rcnn_training_samples["target_labels"], rcnn_training_samples["pred_scores"]
            ),
            "rcnn_reg": self._regression_loss(
                rcnn_training_samples["target_boxes"], rcnn_training_samples["pred_boxes"]
            ),
        }

        nmsed_rpn_output = postprocess_output(self._image_shape, **rpn_output, **self._rpn_config["nms"])
        nmsed_rcnn_output = postprocess_output(self._image_shape, **rcnn_output, **self._rcnn_config["nms"])

        preds = {
            "rpn_boxes": nmsed_rpn_output["pred_boxes"],
            "rpn_scores": nmsed_rpn_output["pred_scores"],
            "rcnn_boxes": nmsed_rcnn_output["pred_boxes"],
            "rcnn_scores": nmsed_rcnn_output["pred_scores"],
            "rcnn_classes": nmsed_rcnn_output["pred_classes"],
        }

        return losses, preds
