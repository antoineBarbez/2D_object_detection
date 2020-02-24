from PIL import ImageDraw, ImageFont

import math

import numpy as np

def draw_box_on_image(image, x_min, y_min, x_max, y_max, label=None, color='red', thickness=2):
	'''
	Draw a box on an image

	Args:
		- image: A PIL Image object.
		- x_min, y_min, x_max, y_max: Coordinates of the bounding box.
		- label: (Optional) Text to add on top of the bounding box.
		- color: (Default: red) Color of the bounding box.
		- thickness: (Default: 2px) Thickness of the line of the bounding box.
	'''
	draw = ImageDraw.Draw(image)

	# Draw bounding box on image
	draw.line(
		[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)],
		width=thickness,
		fill=color)

	# Draw label
	if label is not None:
		font = ImageFont.load_default()
		text_width, text_height = font.getsize(label)
		draw.rectangle(
			[(x_min - math.floor(thickness / 2), y_min - text_height), (x_min + text_width + thickness, y_min)],
			fill=color)
		draw.text(
			(x_min + math.ceil(thickness / 2), y_min - text_height),
			label,
			fill="black",
			font=font)

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
	anchors = np.reshape(anchors, (-1, num_anchors_per_location, 4))

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

		distance_to_im_center = math.sqrt((center_x - im_center_x)**2 + (center_y - im_center_y)**2)
		if distance_to_im_center < best_distance_to_im_center:
			best_distance_to_im_center = distance_to_im_center
			center_location = location

	for anchor in anchors[center_location]:
		x_min, y_min, x_max, y_max = anchor
		draw_box_on_image(image, x_min, y_min, x_max, y_max, color='green')

	draw = ImageDraw.Draw(image)
	draw.point(centers, fill='red')

