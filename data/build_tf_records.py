"""
Creates training and validation TFRecord files from the raw
KITTI detection dataset.
"""
import argparse
import io
import os
from pathlib import Path

import tensorflow as tf
from PIL import Image

import kitti_classes


def parse_args():
    """
    Argument parser.

    Returns:
        Namespace of commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=str, help="Path to images")
    parser.add_argument("--labels-dir", required=True, type=str, help="Path to label files")
    parser.add_argument(
        "--output-dir",
        default="./tf_records",
        type=str,
        help='Path to output TFRecord files, default= "./tf_records". '
        "The training file will be located at: <output_dir>/train.tfrecord "
        "whereas the validation file will be located at: "
        "<output-dir>/valid.tfrecord",
    )
    parser.add_argument(
        "--validation-set-size",
        default=500,
        type=int,
        help="Number of images to be used as a validation set, default=500",
    )
    return parser.parse_args()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_file, label_file):
    with tf.io.gfile.GFile(image_file, "rb") as file:
        encoded_png = file.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size

    name_to_id_map = kitti_classes.get_name_to_id_map()

    ids = []
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    for obj in parse_label_file(label_file):
        if obj["type"] in kitti_classes.class_names:
            ids.append(name_to_id_map[obj["type"]])
            x_mins.append(obj["bbox"][0])
            y_mins.append(obj["bbox"][1])
            x_maxs.append(obj["bbox"][2])
            y_maxs.append(obj["bbox"][3])

    feature = {
        "image/encoded": bytes_feature(encoded_png),
        "image/width": int64_feature(width),
        "image/height": int64_feature(height),
        "label/ids": int64_list_feature(ids),
        "label/x_mins": float_list_feature(x_mins),
        "label/y_mins": float_list_feature(y_mins),
        "label/x_maxs": float_list_feature(x_maxs),
        "label/y_maxs": float_list_feature(y_maxs),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_label_file(label_file):
    with tf.io.gfile.GFile(label_file, "r") as file:
        objects = []
        for line in file:
            fields = line.split()

            obj = {}
            obj["type"] = fields[0]
            obj["truncated"] = float(fields[1])
            obj["occluded"] = int(fields[2])
            obj["alpha"] = float(fields[3])
            obj["bbox"] = list(map(float, fields[4:8]))
            obj["dimensions"] = list(map(float, fields[8:11]))
            obj["location"] = list(map(float, fields[11:14]))
            obj["rotation_y"] = float(fields[14])

            objects.append(obj)

        return objects


def write_tf_records(data_files, output_file):
    output_dir = Path(output_file).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.io.TFRecordWriter(output_file)
    for i, (image_file, label_file) in enumerate(data_files):
        tf_example = create_tf_example(image_file, label_file)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main():
    args = parse_args()

    data_files_train = []
    data_files_valid = []
    for i, image_filename in enumerate(tf.io.gfile.listdir(args.images_dir)):
        image_file = os.path.join(args.images_dir, image_filename)
        label_file = os.path.join(args.labels_dir, Path(image_filename).stem + ".txt")

        if i < args.validation_set_size:
            data_files_valid.append((image_file, label_file))
        else:
            data_files_train.append((image_file, label_file))

    write_tf_records(data_files_train, os.path.join(args.output_dir, "train.tfrecord"))
    write_tf_records(data_files_valid, os.path.join(args.output_dir, "valid.tfrecord"))


if __name__ == "__main__":
    main()
