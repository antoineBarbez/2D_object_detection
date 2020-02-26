import tensorflow as tf
import utils.images as image_utils

import argparse
import datetime
import os
import time

from PIL import Image
from data.input_pipeline import InputPipelineCreator
from models.rpn import RPN

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
		'--logs-dir',
		required=True,
		type=str,
		help='Path to the directory where to write training logs')
	parser.add_argument(
		'--checkpoints-dir',
		required=True,
		type=str,
		help='Path to the directory where to store checkpoint weights')
	parser.add_argument(
		'--num-epochs',
		default=80,
		type=int,
		help='Number of times to go through the data, default=80')
	parser.add_argument(
		'--num-steps-per-epoch',
		default=1000,
		type=int,
		help='number of parameters update per epoch, default=1000')
	return parser.parse_args()

def main():
	num_classes = 7
	image_shape = (375, 1242, 3)
	max_num_objects = 200

	args = parse_args()
	filenames_train = [os.path.join(args.train_data_dir, f) for f in tf.io.gfile.listdir(args.train_data_dir)]
	filenames_valid = [os.path.join(args.valid_data_dir, f) for f in tf.io.gfile.listdir(args.valid_data_dir)]

	pipeline_creator = InputPipelineCreator(
		num_classes=num_classes,
		image_shape=image_shape,
		num_steps_per_epoch=args.num_steps_per_epoch,
		max_num_objects=max_num_objects)
	dataset_train = pipeline_creator.create_input_pipeline(filenames_train)
	dataset_valid = pipeline_creator.create_input_pipeline(filenames_valid, training=False)
	
	train_classification_loss = tf.keras.metrics.Mean(name='train_classification_loss')
	train_regression_loss = tf.keras.metrics.Mean(name='train_regression_loss')

	valid_classification_loss = tf.keras.metrics.Mean(name='valid_classification_loss')
	valid_regression_loss = tf.keras.metrics.Mean(name='valid_regression_loss')

	test_image_path = './data/test_images/000292.png'
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	train_log_dir = os.path.join(args.logs_dir, current_time, 'rpn', 'train')
	valid_log_dir = os.path.join(args.logs_dir, current_time, 'rpn', 'valid')
	image_log_dir = os.path.join(args.logs_dir, current_time, 'rpn', 'image')
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
	image_summary_writer = tf.summary.create_file_writer(image_log_dir)

	learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50000], [0.001, 0.0001])
	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

	model = RPN(image_shape)
	
	with image_summary_writer.as_default():
		image = Image.open(test_image_path)
		image_utils.draw_anchors_on_image(image, model.anchors, 9)
		tf.summary.image('Anchors', image_utils.to_tensor(image), step=0)
		image.close()
	
	for epoch in tf.data.Dataset.range(args.num_epochs):
		for image, classes, boxes in dataset_train:
			objectness = tf.one_hot(tf.ones(max_num_objects, dtype=tf.int32), 2)
			cls_loss, reg_loss = model.train_step(image, objectness, boxes, optimizer)

			train_classification_loss(cls_loss)
			train_regression_loss(reg_loss)

		for image, classes, boxes in dataset_valid:
			objectness = tf.one_hot(tf.ones(max_num_objects, dtype=tf.int32), 2)
			cls_loss, reg_loss = model.test_step(image, objectness, boxes)
			
			valid_classification_loss(cls_loss)
			valid_regression_loss(reg_loss)

		# Update summary
		with train_summary_writer.as_default():
			tf.summary.scalar('classification_loss', train_classification_loss.result(), step=epoch)
			tf.summary.scalar('regression_loss', train_regression_loss.result(), step=epoch)
		
		with valid_summary_writer.as_default():
			tf.summary.scalar('classification_loss', valid_classification_loss.result(), step=epoch)
			tf.summary.scalar('regression_loss', valid_regression_loss.result(), step=epoch)
		
		if (epoch + 1) % 10 == 0:
			# Save model checkpoint
			model.save_weights(os.path.join(args.checkpoints_dir, 'checkpoint_{}'.format(epoch + 1),'weights'), save_format='tf')

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

