import tensorflow as tf

import argparse
import os
import time

from data.input_pipeline import InputPipelineCreator
from models.faster_rcnn.rpn import RPN
from models.faster_rcnn.utils.target_generation import generate_targets

assert tf.__version__.startswith('2')

def parse_args():
	'''
	Argument parser.

	Returns:
		Namespace of commandline arguments
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--train-data-dir',
		required=True,
		type=str,
		help='Path to a directoty containing the TFRecord file(s) for training')
	parser.add_argument(
		'--valid-data-dir',
		required=True,
		type=str,
		help='Path to a directoty containing the TFRecord file(s) for validation')
	parser.add_argument(
		'--num-epochs',
		default=4,
		type=int,
		help='Number of times to go through the data, default=20')
	return parser.parse_args()

def main():
	num_classes = 7
	image_shape = (375, 1242, 3)

	args = parse_args()
	filenames_train = [os.path.join(args.train_data_dir, f) for f in tf.io.gfile.listdir(args.train_data_dir)]
	filenames_valid = [os.path.join(args.valid_data_dir, f) for f in tf.io.gfile.listdir(args.valid_data_dir)]

	pipeline_creator = InputPipelineCreator(num_classes=num_classes, image_shape=image_shape)
	dataset_train = pipeline_creator.create_input_pipeline(filenames_train)
	dataset_valid = pipeline_creator.create_input_pipeline(filenames_valid, shuffle=False, augmentation=False)

	optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

	train_classification_loss = tf.keras.metrics.Mean(name='train_classification_loss')
	train_regression_loss = tf.keras.metrics.Mean(name='train_regression_loss')

	valid_classification_loss = tf.keras.metrics.Mean(name='valid_classification_loss')
	valid_regression_loss = tf.keras.metrics.Mean(name='valid_regression_loss')

	model = RPN(image_shape)

	for epoch in tf.data.Dataset.range(args.num_epochs):
		for image, classes, boxes in dataset_train:
			target_labels, target_boxes = generate_targets(
				gt_boxes=boxes[0], 
				anchor_boxes=model.anchors, 
				image_shape=image_shape,
				num_anchors_per_image=256)
			cls_loss, reg_loss = model.train_step(image, target_labels, target_boxes, optimizer)

			train_classification_loss(cls_loss)
			train_regression_loss(reg_loss)

		for images, classes, boxes in dataset_valid:
			target_labels, target_boxes = generate_targets(
				gt_boxes=boxes[0], 
				anchor_boxes=model.anchors, 
				image_shape=image_shape,
				num_anchors_per_image=256)
			cls_loss, reg_loss = model.test_step(image, target_labels, target_boxes)
			
			valid_classification_loss(cls_loss)
			valid_regression_loss(reg_loss)
		
		# Print metrics of the epoch
		template = 'Epoch {0}/{1}: \n'
		template += '\tTraining   Set --> Classification Loss: {2:.2f}, Regression Loss: {3:.2f}\n'
		template += '\tValidation Set --> Classification Loss: {4:.2f}, Regression Loss: {5:.2f}\n'

		print(template.format(
			epoch + 1,
			args.num_epochs,
			train_classification_loss.result(),
			train_regression_loss.result(),
			valid_classification_loss.result(),
			valid_regression_loss.result()))

		# Reset metric objects
		train_classification_loss.reset_states()
		train_regression_loss.reset_states()
		valid_classification_loss.reset_states()
		valid_regression_loss.reset_states()

if __name__ == "__main__":
	main()

