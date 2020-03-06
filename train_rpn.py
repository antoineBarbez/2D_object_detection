import tensorflow as tf
import utils.images as image_utils
import utils.metrics as metric_utils

import argparse
import datetime
import os

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
		'--num-steps',
		default=80000,
		type=int,
		help='Number of times to go through the data, default=80000')
	parser.add_argument(
		'--num-steps-per-epoch',
		default=500,
		type=int,
		help='Number of steps to complete an epoch, default=500')
	parser.add_argument(
		'--batch-size',
		default=4,
		type=int,
		help='Size of the batches used to update parameters, default=4')
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
		max_num_objects=max_num_objects)
	dataset_train = pipeline_creator.create_input_pipeline(
		filenames=filenames_train, 
		batch_size=args.batch_size, 
		training=True)
	dataset_valid = pipeline_creator.create_input_pipeline(filenames_valid)
	
	train_classification_loss = tf.keras.metrics.Mean(name='train_classification_loss')
	train_regression_loss = tf.keras.metrics.Mean(name='train_regression_loss')
	train_average_precision = metric_utils.AveragePrecision(0.5, name='train_ap_0.5')

	valid_classification_loss = tf.keras.metrics.Mean(name='valid_classification_loss')
	valid_regression_loss = tf.keras.metrics.Mean(name='valid_regression_loss')
	valid_average_precision = metric_utils.AveragePrecision(0.5, name='valid_ap_0.5')

	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	train_log_dir = os.path.join(args.logs_dir, current_time, 'rpn', 'train')
	valid_log_dir = os.path.join(args.logs_dir, current_time, 'rpn', 'valid')
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

	learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50000], [0.001, 0.0001])
	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

	model = RPN(image_shape)

	for step, (images, classes, boxes) in dataset_train.enumerate():
		cls_loss, reg_loss, pred_scores, pred_boxes = model.train_step(images, classes, boxes, optimizer)

		train_classification_loss(cls_loss)
		train_regression_loss(reg_loss)
		train_average_precision(boxes, pred_boxes, pred_scores)

		if (step + 1) % args.num_steps_per_epoch == 0:
			epoch = (step + 1) // args.num_steps_per_epoch

			with train_summary_writer.as_default():
				tf.summary.scalar('Losses/classification_loss', train_classification_loss.result(), step=step + 1)
				tf.summary.scalar('Losses/regression_loss', train_regression_loss.result(), step=step + 1)
				tf.summary.scalar('Metrics/average_precision', train_average_precision.result(), step=step + 1)

			with valid_summary_writer.as_default():
				for test_step, (images, classes, boxes) in dataset_valid.enumerate():
					cls_loss, reg_loss, pred_scores, pred_boxes, foreground_regions, background_regions = model.test_step(images, classes, boxes)

					valid_classification_loss(cls_loss)
					valid_regression_loss(reg_loss) 
					valid_average_precision(boxes, pred_boxes, pred_scores)

					if test_step == 2:
						# Add ground-truth boxes and anchors to summary only once 
						if epoch == 1:
							image_gt_boxes = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
							image_utils.draw_predictions_on_image(image_gt_boxes, boxes[0], relative=True)
							tf.summary.image('Ground-truth', image_utils.to_tensor(image_gt_boxes), step=0)
							image_gt_boxes.close()

							image_anchors_resume = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
							image_utils.draw_anchors_on_image(image_anchors_resume, model.detector.anchors, 12)
							tf.summary.image('Anchors/resume', image_utils.to_tensor(image_anchors_resume), step=0)
							image_anchors_resume.close()

							image_anchors_foreground = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
							image_utils.draw_predictions_on_image(image_anchors_foreground, foreground_regions, relative=False)
							tf.summary.image('Anchors/foreground', image_utils.to_tensor(image_anchors_foreground), step=0)
							image_anchors_foreground.close()

							image_anchors_background = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
							image_utils.draw_predictions_on_image(image_anchors_background, background_regions, relative=False)
							tf.summary.image('Anchors/background', image_utils.to_tensor(image_anchors_background), step=0)
							image_anchors_background.close()

						# Add predictions to summary every 5 epochs
						if epoch % 5 == 0:
							image_predictions = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
							rois = tf.gather_nd(pred_boxes, tf.where(pred_scores > 0.5))
							image_utils.draw_predictions_on_image(image_predictions, rois, relative=True)
							tf.summary.image('Predictions', image_utils.to_tensor(image_predictions), step=step + 1)
							image_predictions.close()

				tf.summary.scalar('Losses/classification_loss', valid_classification_loss.result(), step=step + 1)
				tf.summary.scalar('Losses/regression_loss', valid_regression_loss.result(), step=step + 1)
				tf.summary.scalar('Metrics/average_precision', valid_average_precision.result(), step=step + 1)

			# Print metrics of the epoch
			template = 'Epoch {0}/{1}: \n'
			template += '\tCls Loss --> Train: {2:.2f}, Valid: {3:.2f}\n'
			template += '\tReg Loss --> Train: {4:.2f}, Valid: {5:.2f}\n'
			template += '\tAP       --> Train: {6:.2f}, Valid: {7:.2f}\n'

			tf.print(template.format(
				epoch,
				args.num_steps // args.num_steps_per_epoch,
				train_classification_loss.result(),
				valid_classification_loss.result(),
				train_regression_loss.result(),
				valid_regression_loss.result(),
				train_average_precision.result(),
				valid_average_precision.result()))

			# Reset metric objects
			train_classification_loss.reset_states()
			train_regression_loss.reset_states()
			train_average_precision.reset_states()
			valid_classification_loss.reset_states()
			valid_regression_loss.reset_states()
			valid_average_precision.reset_states()

			# Save model checkpoint every 10 epochs
			if epoch % 10 == 0:
				model.save_weights(os.path.join(args.checkpoints_dir, 'checkpoint_{}'.format(epoch),'weights'), save_format='tf')

		if (step + 1) == args.num_steps:
			break

if __name__ == "__main__":
	main()

