import io
import math

import seaborn as sns
import tensorflow as tf
from PIL import ImageDraw, ImageFont

from utils.boxes import to_absolute


def draw_box_on_image(image, box, label=None, relative=False, color="red", thickness=2):
    """
    Draw a box on an image

    Args:
        - image: A PIL Image object.
        - box: Coordinates of the bounding box.
        - label: (Optional) Text to add on top of the bounding box.
        - relative: Whether the boxe is in relative or absolute coordinates.
        - color: (Default: red) Color of the bounding box.
        - thickness: (Default: 2px) Thickness of the line of the bounding box.
    """
    # Draw bounding box on image
    if relative:
        width, height = image.size
        box = to_absolute(box, [height, width, 3])

    x_min, y_min, x_max, y_max = box

    draw = ImageDraw.Draw(image)
    draw.line(
        [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)], width=thickness, fill=color
    )

    # Draw label
    if label is not None:
        font = ImageFont.load_default()
        text_width, text_height = font.getsize(label)
        draw.rectangle(
            [(x_min - math.floor(thickness / 2), y_min - text_height), (x_min + text_width + thickness, y_min)],
            fill=color,
        )
        draw.text((x_min + math.ceil(thickness / 2), y_min - text_height), label, fill="black", font=font)


def draw_predictions_on_image(
    image, boxes, scores=None, class_indices=None, class_names=None, relative=False, default_color="red", thickness=2
):
    """
    Draw predicted boxes with scores on image.

    Args:
        - image: A PIL Image object.
        - boxes: A tensor of shape [num_pred, 4].
        - scores: A tensor of shape [num_pred].
        - class_indices: A tensor of shape [num_pred]
        - class_names: A list containing the name of the classes.
        - relative: Whether the boxes are in relative or absolute coordinates.
        - default_color: (Default: 'red') Color to use for boxes if class_scores and
            class_names are not specified.
        - thickness: (Default: 2px) Thickness of the line of the bounding box.
    """
    if (class_names is not None) & (class_indices is not None):
        color_palette = sns.color_palette("hls", len(class_names))
        color_palette = [tuple(int(channel * 255) for channel in color) for color in color_palette]

    for i, box in enumerate(boxes):
        label = None
        color = default_color

        if (class_names is not None) & (class_indices is not None):
            class_indice = class_indices[i]
            label = class_names[class_indice]
            color = color_palette[class_indice]

        if scores is not None:
            if label is None:
                label = ""
            else:
                label += ": "
            label += "{:.0f}%".format(scores[i] * 100)

        draw_box_on_image(image=image, box=box, label=label, relative=relative, color=color, thickness=thickness)


def to_tensor(image):
    """
    Args:
        - image: A PIL Image object.

    Returns:
        A tensor of shape [1, heigth, width, 3]
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image = buf.getvalue()
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.io.decode_image(image, channels=3, dtype=tf.float32)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image
