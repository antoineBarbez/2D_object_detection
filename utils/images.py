from PIL import ImageDraw, ImageFont

import io
import math

import utils.boxes as box_utils
import seaborn as sns
import tensorflow as tf

def draw_box_on_image(image, box, label=None, relative=False, color='red', thickness=2):
	'''
	Draw a box on an image

	Args:
		- image: A PIL Image object.
		- box: Coordinates of the bounding box.
		- label: (Optional) Text to add on top of the bounding box.
		- relative: Whether the boxe is in relative or absolute coordinates.
		- color: (Default: red) Color of the bounding box.
		- thickness: (Default: 2px) Thickness of the line of the bounding box.
	'''
	# Draw bounding box on image
	if relative:
		width, height = image.size
		box = box_utils.to_absolute(box, [height, width, 3])

	x_min, y_min, x_max, y_max = box

	draw = ImageDraw.Draw(image)
	draw.line(
		[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)],
		width=thickness,
		fill=color)

	# Draw label
	if label is not None:
		font = ImageFont.load_default()
		text_width, text_height = font.getsize(label)
		draw.rectangle(
			[(x_min - math.floor(thickness / 2), y_min - text_height),
			 (x_min + text_width + thickness, y_min)],
			fill=color)
		draw.text(
			(x_min + math.ceil(thickness / 2), y_min - text_height),
			label,
			fill="black",
			font=font)

def draw_predictions_on_image(image, boxes, class_scores=None, 
	class_names=None, relative=False, default_color='red', thickness=2):
	'''
	Draw predicted boxes with scores on image.

	Args:
		- image: A PIL Image object.
		- boxes: A tensor of shape [num_pred, 4].
		- class_scores: A tensor of shape [num_pred, num_classes].
		- class_names: A list of size num_classes.
		- relative: Whether the boxes are in relative or absolute coordinates.
		- default_color: (Default: 'red') Color to use for boxes if class_scores and 
			class_names are not specified.
		- thickness: (Default: 2px) Thickness of the line of the bounding box.
	'''
	if (class_scores is not None) & (class_names is not None):
		color_palette = sns.color_palette("hls", len(class_names))
		class_indices = tf.math.argmax(class_scores, -1)

	for i, box in enumerate(boxes):
		if (class_scores is not None) & (class_names is not None):
			class_indice = class_indices[i]
			label = "{0}: {1:.2f}".format(
				class_names[class_indice],
				class_scores[i, class_indice])
			color = color_palette[class_indice]
		else:
			label = None
			color = default_color
		
		draw_box_on_image(
			image=image, 
			box=box,
			label=label,
			relative=relative,
			color=color,
			thickness=thickness)

def draw_anchors_on_image(image, anchors, num_anchors_per_location):
	'''
	Draw anchor boxes on an image.

	This function draws the center of each anchors on the image, and
	only num_anchors_per_location anchors at the center location.

	Args:
		- image: A PIL Image object.
		- anchors: A numpy.Array of shape [N, 4].
		- num_anchors_per_location: Number of anchors at each location in the image. 
	'''
	anchors = tf.reshape(anchors, (-1, num_anchors_per_location, 4))

	im_width, im_height = image.size
	im_center_x = im_width / 2.0
	im_center_y = im_height / 2.0

	centers = []
	best_distance_to_im_center = 10000.0
	center_location = 0
	for location, anchors_at_location in enumerate(anchors):
		x_min, y_min, x_max, y_max = anchors_at_location[0]
		center_x = (x_min + x_max) / 2.0
		center_y = (y_min + y_max) / 2.0
		centers.append((center_x, center_y))

		distance_to_im_center = math.sqrt(
			(center_x - im_center_x)**2 + 
			(center_y - im_center_y)**2)
		if distance_to_im_center < best_distance_to_im_center:
			best_distance_to_im_center = distance_to_im_center
			center_location = location

	draw_predictions_on_image(image, anchors[center_location], default_color='green')

	draw = ImageDraw.Draw(image)
	draw.point(centers, fill='red')

def to_tensor(image):
	'''
	Args:
		- image: A PIL Image object.

	Returns:
		A tensor of shape [1, heigth, width, 3]
	'''
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	image = buf.getvalue()
	buf.seek(0)
	
	# Convert PNG buffer to TF image
	image = tf.io.decode_image(image, channels=3, dtype=tf.float32)
	
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	
	return image

