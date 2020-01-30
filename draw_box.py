from PIL import ImageDraw, ImageFont

import math

def draw_bbox_on_image(image, x_min, y_min, x_max, y_max, label=None, color='red', thickness=2):
	'''
	Draw a bounding box on an image

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