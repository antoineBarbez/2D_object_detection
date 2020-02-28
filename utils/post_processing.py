import tensorflow as tf
import utils.boxes as box_utils

def postprocess_output(self,
		regions,
		image_shape,
		pred_class_scores,
		pred_boxes_encoded,
		max_output_size_per_class,
		max_total_size,
		iou_threshold):
		'''
		Args:
			- regions: A tensor of shape [num_regions, 4] representing the reference boxes
				to be used for decoding predicted boxes.
			- image_shape: Shape of the image to be used for clipping and normalization.
			- pred_class_scores: Output of the classification head. A tensor of shape 
				[batch_size, num_regions, num_classes + 1] representing classification scores
				with background.
			- pred_boxes_encoded: Output of the regression head. A tensor of shape
				[batch_size, num_regions, num_classes, 4] representing encoded predicted box coordinates.
			- max_output_size_per_class: A scalar integer Tensor representing the maximum number of boxes 
				to be selected by non max suppression per class.
			- max_total_size: A scalar representing maximum number of boxes retained over all classes.
			- iou_threshold: A float representing the threshold for deciding whether boxes overlap 
				too much with respect to IOU.

		Returns:
			- nmsed_boxes: A [batch_size, max_total_size, 4] float32 tensor containing the non-max suppressed boxes.
			- nmsed_scores: A [batch_size, max_total_size] float32 tensor containing the scores for the boxes.
			- nmsed_classes: A [batch_size, max_total_size] float32 tensor containing the class for boxes.
			- num_valid_detections: A [batch_size] int32 tensor indicating the number of valid 
				detections per batch item. The rest of the entries are zero paddings.
		'''
		pred_boxes = box_utils.decode(pred_boxes_encoded, regions)

		# Clip boxes to image boundaries
		image_height, image_width, _ = image_shape
		pred_boxes = box_utils.clip_to_window(pred_boxes, [0, 0, image_width, image_height])

		# Non Max Suppression
		pred_class_scores = pred_class_scores[..., 1:]
		(nmsed_boxes, nmsed_scores, nmsed_classes, num_valid_detections) = (
			tf.image.combined_non_max_suppression(
				boxes=pred_boxes,
				scores=pred_class_scores,
				max_output_size_per_class=max_output_size_per_class,
				max_total_size=max_total_size,
				iou_threshold=iou_threshold,
				score_threshold=0.0))

		# TODO
		# Normalization

		return nmsed_boxes, nmsed_scores, nmsed_classes, num_valid_detections