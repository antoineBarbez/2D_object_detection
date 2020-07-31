import tensorflow as tf

def sample_image(
	target_class_labels,
	target_boxes_encoded,
	pred_class_scores,
	pred_boxes_encoded,
	regions,
	num_samples,
	foreground_proportion):
	''' 
	Args:
		- target_class_labels: A tensor of shape [num_regions, num_classes + 1] representing the target
			labels for each region. The target label for ignored regions is zeros(num_classes + 1).
		- target_boxes_encoded: A tensor of shape [num_regions, num_classes, 4] representing the encoded target ground-truth 
			bounding box for each region, i.e., the ground-truth bounding box with the highest IoU 
			overlap with the considered region. 
		- pred_class_scores: Output of the classification head. A tensor of shape 
			[num_regions, num_classes + 1] representing classification scores.
		- pred_boxes_encoded: Output of the regression head. A tensor of shape
			[num_regions, num_classes, 4] representing encoded predicted box coordinates.
		- num_samples: Number of examples (regions) to sample per image.
		- foreground_proportion: Maximum proportion of foreground vs background  
			examples to sample in each minibatch. This parameter is set to 0.5 for
			the RPN and 0.25 for the Faster-RCNN according to the respective papers.

	Returns:
		sampled target_class_labels, target_boxes_encoded, pred_class_scores, and 
		pred_boxes_encoded. The first dimension of these tensors equals num_samples.
	'''

	foreground_inds = tf.reshape(tf.where(
		(tf.reduce_sum(target_class_labels, -1) != 0.0) &
		(target_class_labels[:, 0] == 0.0)), [-1])

	background_inds = tf.reshape(tf.where(
		(tf.reduce_sum(target_class_labels, -1) != 0.0) &
		(target_class_labels[:, 0] == 1.0)), [-1])

	num_foreground_inds = tf.minimum(
		tf.size(foreground_inds),
		tf.cast(tf.math.round(num_samples*foreground_proportion), dtype=tf.int32))
	num_background_inds = num_samples - num_foreground_inds
	
	foreground_inds = tf.random.shuffle(foreground_inds)
	foreground_inds = foreground_inds[:num_foreground_inds]
	
	background_inds = tf.gather(background_inds, tf.random.uniform([num_background_inds], minval=0, maxval=tf.size(background_inds), dtype=tf.int32))

	inds_to_keep = tf.concat([foreground_inds, background_inds], 0)
	inds_to_keep.set_shape([num_samples])

	target_class_labels_sample = tf.gather(target_class_labels, inds_to_keep)
	pred_class_scores_sample = tf.gather(pred_class_scores, inds_to_keep)
	target_boxes_encoded_sample = tf.gather(target_boxes_encoded, inds_to_keep)
	pred_boxes_encoded_sample = tf.gather(pred_boxes_encoded, inds_to_keep)
	regions_sample = tf.gather(regions, inds_to_keep)

	return (target_class_labels_sample, target_boxes_encoded_sample, 
		pred_class_scores_sample, pred_boxes_encoded_sample, regions_sample)