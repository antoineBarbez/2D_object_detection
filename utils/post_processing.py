import tensorflow as tf

from utils.boxes import decode, to_relative


def postprocess_output(
    image_shape,
    regions,
    pred_scores,
    pred_boxes,
    score_threshold,
    iou_threshold,
    max_output_size_per_class,
    max_total_size,
):
    """
        Args:
            - image_shape: Shape of the image to be used for clipping and normalization.
            - regions: Tensor of shape [batch_size, num_regions, 4] or [num_regions, 4],
                representing the reference regions (RoIs for Faster-RCNN or anchor boxes for RPN).
            - pred_scores: Tensor of shape [batch_size, num_regions, num_classes +1].
                Output of the classification head, representing classification scores.
            - pred_boxes: Tensor of shape [batch_size, num_regions, num_classes, 4].
                Output of the regression head, representing encoded predicted boxes.
            - score_threshold: A float representing the threshold for deciding when to remove boxes based on score.
            - iou_threshold: A float representing the threshold for deciding whether boxes overlap
                too much with respect to IOU.
            - max_output_size_per_class: A scalar integer Tensor representing the maximum number of boxes
                to be selected by non max suppression per class.
            - max_total_size: A scalar representing maximum number of boxes retained over all classes.

        Returns:
            - nmsed_boxes: A [batch_size, max_total_size, 4] float32 tensor containing the non-max suppressed boxes.
            - nmsed_scores: A [batch_size, max_total_size] float32 tensor containing the scores for the boxes.
            - nmsed_classes: A [batch_size, max_total_size] int32 tensor containing the class indices for the boxes.
            - num_valid_detections: A [batch_size] int32 tensor indicating the number of valid
                detections per batch item. The rest of the entries are zero paddings.
    """
    # Tile regions to be broadcastable with pred_boxes_encoded for decoding
    num_classes = tf.shape(pred_boxes)[2]
    regions = tf.expand_dims(regions, -2)
    multiples = tf.one_hot(tf.rank(regions) - 2, tf.rank(regions), on_value=num_classes, off_value=1, dtype=tf.int32)
    regions = tf.tile(regions, multiples)

    # Decode predicted boxes
    pred_boxes = decode(pred_boxes, regions)

    # Normalize predicted boxes
    pred_boxes = to_relative(pred_boxes, image_shape)

    # Non Max Suppression and clipping
    pred_scores = pred_scores[..., 1:]
    (nmsed_boxes, nmsed_scores, nmsed_classes, num_valid_detections) = tf.image.combined_non_max_suppression(
        pred_boxes, pred_scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold
    )

    nmsed_output = {}
    nmsed_output["pred_boxes"] = tf.stop_gradient(nmsed_boxes)
    nmsed_output["pred_scores"] = tf.stop_gradient(nmsed_scores)
    nmsed_output["pred_classes"] = tf.stop_gradient(tf.cast(nmsed_classes, dtype=tf.int32))
    nmsed_output["num_valid_detections"] = tf.stop_gradient(num_valid_detections)

    return nmsed_output
