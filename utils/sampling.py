import tensorflow as tf


def sample_image(target_class_labels, num_samples, foreground_proportion):
    """
    Args:
        - target_class_labels: A tensor of shape [num_regions, num_classes + 1] representing the target
            labels (with background) for each region.
            The target label for ignored regions is zeros(num_classes + 1).
        - num_samples: Number of examples (regions) to sample per image.
        - foreground_proportion: Maximum proportion of foreground vs background
            examples to sample in each minibatch. This parameter is set to 0.5 for
            the RPN and 0.25 for the Faster-RCNN according to the respective papers.

    Returns:
        Tensor of shape [num_samples] containing the indices of the regions to be taken into account for
        computing the losses.

    """

    foreground_inds = tf.reshape(
        tf.where((tf.reduce_sum(target_class_labels, -1) != 0.0) & (target_class_labels[:, 0] == 0.0)), [-1]
    )

    background_inds = tf.reshape(
        tf.where((tf.reduce_sum(target_class_labels, -1) != 0.0) & (target_class_labels[:, 0] == 1.0)), [-1]
    )

    num_foreground_inds = tf.minimum(
        tf.size(foreground_inds), tf.cast(tf.math.round(num_samples * foreground_proportion), dtype=tf.int32)
    )
    num_background_inds = num_samples - num_foreground_inds

    foreground_inds = tf.random.shuffle(foreground_inds)
    foreground_inds = foreground_inds[:num_foreground_inds]

    background_inds = tf.gather(
        background_inds,
        tf.random.uniform([num_background_inds], minval=0, maxval=tf.size(background_inds), dtype=tf.int32),
    )

    inds_to_keep = tf.concat([foreground_inds, background_inds], 0)
    inds_to_keep.set_shape([num_samples])
    return inds_to_keep
