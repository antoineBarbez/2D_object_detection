import tensorflow as tf

import argparse
import os
import time

from data.input_pipeline import InputPipelineCreator
from models.faster_rcnn.rpn import RPN

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
		default=3,
		type=int,
		help='Number of times to go through the data, default=20')
	parser.add_argument(
		'--batch-size',
		default=2,
		type=int,
		help='Number of training examples to process at each learning step, default=8')
	return parser.parse_args()

def main():
	num_classes = 7
	image_shape = (375, 1242, 3)

	args = parse_args()
	filenames_train = [os.path.join(args.train_data_dir, f) for f in tf.io.gfile.listdir(args.train_data_dir)]
	#filenames_valid = [os.path.join(args.valid_data_dir, f) for f in tf.io.gfile.listdir(args.valid_data_dir)]

	pipeline_creator = InputPipelineCreator(num_classes=num_classes, image_shape=image_shape)
	dataset_train = pipeline_creator.create_input_pipeline(filenames_train, args.batch_size)
	#dataset_valid = pipeline_creator.create_input_pipeline(filenames_valid, 1, shuffle=False, augmentation=False)

	optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

	train_classification_loss = tf.keras.metrics.Mean(name='train_classification_loss')
	train_regression_loss = tf.keras.metrics.Mean(name='train_regression_loss')

	model = RPN(image_shape)

	@tf.function
	def train_step(images, gt_boxes):
		with tf.GradientTape() as tape:
			pr_objectness, pr_boxes = model(images, training=True)
			
			cls_loss, reg_loss = model.get_losses(gt_boxes, pr_objectness, pr_boxes)
			multi_task_loss = cls_loss + reg_loss

		gradients = tape.gradient(multi_task_loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_classification_loss(cls_loss)
		train_regression_loss(reg_loss)

	for epoch in tf.data.Dataset.range(args.num_epochs):
		for images, classes, boxes in dataset_train:
			train_step(images, boxes)
		
		# Print metrics of the epoch
		template = 'Classification Loss: {0:.2f}, Regression Loss: {1:.2f}'
		print(template.format(
			train_classification_loss.result(),
			train_regression_loss.result()))

		# Reset metric objects
		train_classification_loss.reset_states()
		train_regression_loss.reset_states()

if __name__ == "__main__":
	main()

