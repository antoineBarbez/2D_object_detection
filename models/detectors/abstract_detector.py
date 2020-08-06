import tensorflow as tf

from abc import abstractmethod


class AbstractDetector(tf.keras.Model):
    """
    Abstract base class for object detectors

    Here we call detector a model that outputs predictions from feature maps,
    i.e., the output of a feature extractor.
    """

    def __init__(self, **kwargs):
        super(AbstractDetector, self).__init__(**kwargs)

    @abstractmethod
    def call(self, feature_maps, **kwargs):
        """
        Args:
            - feature_maps: Output of the feature extractor.

        Returns:
            - regions: Reference boxes to be used for decoding the output of the regression head.
                These correspond to the anchor boxes for the RPN model and to the RoI's for the
                Fast-RCNN model. shape = [num_regions, 4].
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_regions, num_classes + 1] representing classification scores.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape
                [batch_size, num_regions, num_classes, 4] representing encoded 
                predicted box coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess_output(self, regions, pred_class_scores, pred_boxes_encoded, **kwargs):
        """
        Postprocess the output of the object detector.

        Args:
            - regions: Reference boxes to be used for decoding the output of the regression head.
                These correspond to the anchor boxes for the RPN model and to the RoI's for the
                Fast-RCNN model. shape = [num_regions, 4].
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_regions, num_classes + 1] representing classification scores.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape
                [batch_size, num_regions, num_classes, 4] representing encoded 
                predicted box coordinates.
        """
        raise NotImplementedError
