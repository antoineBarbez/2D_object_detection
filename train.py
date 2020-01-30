import tensorflow as tf

import argparse
import os
import time

from data.input_pipeline import InputPipelineCreator

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
		'--num_epochs',
		default=3,
		type=int,
		help='Number of times to go through the data, default=20')
	parser.add_argument(
		'--batch_size',
		default=10,
		type=int,
		help='Number of training examples to process at each learning step, default=10')
	return parser.parse_args()

def main():
	args = parse_args()
	filenames_train = [os.path.join(args.train_data_dir, f) for f in tf.io.gfile.listdir(args.train_data_dir)]
	filenames_valid = [os.path.join(args.valid_data_dir, f) for f in tf.io.gfile.listdir(args.valid_data_dir)]

	pipeline_creator = InputPipelineCreator(num_classes=7, image_shape=[375, 1242, 3])
	dataset_train = pipeline_creator.create_input_pipeline(filenames_train, args.batch_size)
	dataset_valid = pipeline_creator.create_input_pipeline(filenames_valid, 1, shuffle=False, augmentation=False)
	
	for epoch in tf.data.Dataset.range(args.num_epochs):
		start_time_train = time.perf_counter()
		for images, classes, bboxes in dataset_train:
			pass
		training_time = time.perf_counter() - start_time_train
		
		start_time_valid = time.perf_counter()
		for image, classes, bboxes	in dataset_valid:
			pass
		validation_time = time.perf_counter() - start_time_valid
		tf.print('Epoch {}: Training time = {}, Validation time = {}'.format(epoch, training_time, validation_time))

if __name__ == "__main__":
	main()

