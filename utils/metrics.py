import tensorflow as tf


class AveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold, num_points=11, **kwargs):
        super(AveragePrecision, self).__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.num_points = num_points

        self.true_count = self.add_weight(name="true_count", dtype=tf.int32, initializer="zeros")
        self.pos_count = self.add_weight(name="pos_count", dtype=tf.int32, initializer="zeros")

        self.true_pos = tf.Variable(
            [], shape=tf.TensorShape(None), dtype=tf.int32, synchronization=tf.VariableSynchronization.ON_READ
        )
        self.scores = tf.Variable(
            [], shape=tf.TensorShape(None), dtype=tf.float32, synchronization=tf.VariableSynchronization.ON_READ
        )

    def reset_states(self):
        super(AveragePrecision, self).reset_states()
        self.true_pos.assign([])
        self.scores.assign([])

    def result(self):
        sorted_true_pos = tf.gather(self.true_pos, tf.argsort(self.scores, direction="DESCENDING"))
        cumulative_true_pos = tf.cast(tf.cumsum(sorted_true_pos), dtype=tf.float32)
        cumulative_pos = tf.cast(tf.range(self.pos_count) + 1, dtype=tf.float32)

        precisions = cumulative_true_pos / cumulative_pos
        recalls = cumulative_true_pos / tf.cast(self.true_count, dtype=tf.float32)

        precisions = tf.concat([precisions, tf.constant([0.0])], axis=0, name="con_1")
        recalls = tf.concat([recalls, tf.constant([2.0])], axis=0, name="con_2")

        recall_values = tf.range(self.num_points, dtype=tf.float32) / tf.cast(self.num_points - 1, dtype=tf.float32)
        interpolated_precisions = tf.map_fn(
            fn=lambda r: tf.reduce_max(tf.gather(precisions, tf.where(recalls >= r))), elems=recall_values,
        )

        return tf.reduce_mean(interpolated_precisions)

    def update_state(self, gt_boxes, pred_boxes, pred_scores):
        """
        Args:
            - gt_boxes: A tensor of shape [batch_size, max_num_objects, 4] possibly zero padded
                representing the ground-truth boxes.
            - pred_boxes: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded
                representing predicted bounding-boxes.
            - pred_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded
                representing the scores for each predicted box.
        """
        batch_size = tf.shape(pred_boxes)[0]
        for i in tf.range(batch_size):
            self.update_state_single_image(gt_boxes=gt_boxes[i], pred_boxes=pred_boxes[i], pred_scores=pred_scores[i])

    def update_state_single_image(self, gt_boxes, pred_boxes, pred_scores):
        """
        Args:
            - gt_boxes: A tensor of shape [max_num_objects, 4] possibly zero padded
                representing the ground-truth boxes.
            - pred_boxes: A tensor of shape [max_predictions, 4] possibly zero padded
                representing predicted bounding-boxes.
            - pred_scores: A tensor of shape [max_predictions] possibly zero padded
                representing the scores for each predicted box.
        """
        # Remove padding ground-truth boxes
        gt_non_padding_inds = tf.where(tf.reduce_sum(gt_boxes, -1) != 0.0)
        gt_boxes = tf.gather_nd(gt_boxes, gt_non_padding_inds)

        # Update state variables
        num_gt_boxes = tf.shape(gt_boxes)[0]
        num_pred_boxes = tf.shape(pred_boxes)[0]

        self.true_count.assign_add(num_gt_boxes)
        self.pos_count.assign_add(num_pred_boxes)
        self.scores.assign(tf.concat([self.scores, pred_scores], axis=0, name="con_3"))

        ious = iou(gt_boxes, pred_boxes, pairwise=True)
        ious = ious * tf.one_hot(tf.math.argmax(ious, axis=1), num_pred_boxes)
        true_positives = tf.where(ious > self.iou_threshold, 1.0, 0.0)
        true_positives = tf.reduce_sum(true_positives, axis=0)
        true_positives = tf.cast(true_positives > 0.0, dtype=tf.int32)
        self.true_pos.assign(tf.concat([self.true_pos, true_positives], axis=0, name="con_4"))


class MeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes, iou_threshold, num_points=11, **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.num_classes = num_classes

        self.average_precisions = [AveragePrecision(iou_threshold, num_points) for _ in range(num_classes)]

    def reset_states(self):
        for average_precision in self.average_precisions:
            average_precision.reset_states()

    def result(self):
        sum_ap = 0.0
        for average_precision in self.average_precisions:
            sum_ap += average_precision.result()

        return sum_ap / float(self.num_classes)

    def update_state(self, gt_boxes, gt_class_labels, pred_boxes, pred_scores, pred_classes):
        """
        Args:
            - gt_boxes: A tensor of shape [batch_size, max_num_objects, 4] possibly zero padded
                representing the ground-truth boxes.
            - gt_class_labels: [batch_size, max_num_objects, num_classes + 1] possibly zero padded
                representing the ground-truth class labels with background.
            - pred_boxes: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded
                representing predicted bounding-boxes.
            - pred_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded
                representing the class score for each predicted box.
            - pred_classes: A tensor of shape [batch_size, max_predictions] possibly zero padded
                representing the class indice for each predicted box.
        """
        gt_class_labels = gt_class_labels[..., 1:]
        for i in range(self.num_classes):
            self.average_precisions[i].update_state(
                gt_boxes=tf.where(
                    condition=tf.tile(input=tf.expand_dims(gt_class_labels[:, :, i] == 1.0, -1), multiples=[1, 1, 4]),
                    x=gt_boxes,
                    y=0.0,
                ),
                pred_boxes=tf.where(
                    condition=tf.tile(input=tf.expand_dims(pred_classes == i, -1), multiples=[1, 1, 4]),
                    x=pred_boxes,
                    y=0.0,
                ),
                pred_scores=tf.where(condition=pred_classes == i, x=pred_scores, y=0.0),
            )


def area(boxes):
    """
    Computes area of boxes.

    Args:
        - boxes: A tf.float32 tensor of shape [N, 4].

    Returns:
        A tf.float32 tensor of shape [N].
    """
    x_mins, y_mins, x_maxs, y_maxs = tf.split(boxes, num_or_size_splits=4, axis=-1)
    return tf.squeeze((x_maxs - x_mins) * (y_maxs - y_mins), [-1])


def intersection(boxes_1, boxes_2, pairwise=False):
    """
    Compute pairwise intersection areas between boxes.

    Args:
        - boxes_1: A tf.float32 tensor of shape [N, 4].
        - boxes_2: A tf.float32 tensor of shape [M, 4].
        - pairwise: A boolean, if True this operation returns the pairwise
            intersection values else, it returns the elementwise intersections.

    Returns:
        A tf.float32 tensor of shape [N, M] if pairwise is True
        else, a tf.float32 tensor of shape [N].
    """
    x_mins_1, y_mins_1, x_maxs_1, y_maxs_1 = tf.split(boxes_1, num_or_size_splits=4, axis=-1)
    x_mins_2, y_mins_2, x_maxs_2, y_maxs_2 = tf.split(boxes_2, num_or_size_splits=4, axis=-1)

    if pairwise:
        x_mins_2 = tf.transpose(x_mins_2)
        y_mins_2 = tf.transpose(y_mins_2)
        x_maxs_2 = tf.transpose(x_maxs_2)
        y_maxs_2 = tf.transpose(y_maxs_2)

    diff_widths = tf.minimum(x_maxs_1, x_maxs_2) - tf.maximum(x_mins_1, x_mins_2)
    diff_heights = tf.minimum(y_maxs_1, y_maxs_2) - tf.maximum(y_mins_1, y_mins_2)
    intersections = tf.maximum(0.0, diff_widths) * tf.maximum(0.0, diff_heights)

    if pairwise:
        return intersections
    else:
        return tf.reshape(intersections, [-1])


def iou(boxes_1, boxes_2, pairwise=False):
    """
    Computes intersection-over-union between two set of boxes.

    Args:
        - boxes_1: A tf.float32 tensor of shape [N, 4].
        - boxes_2: A tf.float32 tensor of shape [M, 4].
        - pairwise: A boolean, if True this operation returns the pairwise
            iou scores else, it returns the elementwise iou scores.

    Returns:
        A tf.float32 tensor of shape [N, M] if pairwise is True
        else, a tf.float32 tensor of shape [N].
    """

    intersections = intersection(boxes_1, boxes_2, pairwise)

    areas_1 = area(boxes_1)
    areas_2 = area(boxes_2)
    if pairwise:
        areas_1 = tf.expand_dims(areas_1, 1)
        areas_2 = tf.expand_dims(areas_2, 0)

    unions = areas_1 + areas_2 - intersections

    return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections), tf.truediv(intersections, unions))
