import tensorflow as tf

import argparse
import os
import time

from data.input_pipeline import InputPipeline

assert tf.__version__.startswith('2')

def parse_args():
	'''
	Argument parser.

	Returns:
		Namespace of commandline arguments
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		required=True,
		type=str,
		help='Path to TFRecord data files')
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
	filenames = [os.path.join(args.data_dir, f) for f in tf.io.gfile.listdir(args.data_dir)]

	pipeline = InputPipeline(num_class=7, image_shape=[375, 1242, 3])
	dataset_train = pipeline.get_input_pipeline(filenames, args.batch_size)
	for epoch in tf.data.Dataset.range(args.num_epochs):
		start_time = time.perf_counter()
		count = 0
		for image, classes, bboxes in dataset_train:
			count = count + 1
			pass

		tf.print('epoch {}: Count = {}, time = {}'.format(epoch, count, time.perf_counter() - start_time))

if __name__ == "__main__":
	main()

