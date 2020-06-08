import tensorflow as tf
import utils.images as image_utils
import utils.metrics as metric_utils

import argparse
import os

from PIL import Image
from data.input_pipeline import InputPipelineCreator
from data.kitti_classes import class_names
from models.faster_rcnn import FasterRCNN

assert tf.__version__.startswith("2")


def parse_args():
    """
    Argument parser.

    Returns:
        Namespace of commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", required=True, type=str, help="Path to the training TFRecord file",
    )
    parser.add_argument(
        "--valid-data-path", required=True, type=str, help="Path to the validation TFRecord file",
    )
    parser.add_argument(
        "--logs-dir", default="logs", type=str, help="Path to the directory where to write training logs",
    )
    parser.add_argument(
        "--save-dir",
        default="saved_models",
        type=str,
        help="Path to the directory where to store weights of the final model",
    )
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints", type=str, help="Path to the directory where to store checkpoints",
    )
    parser.add_argument("--num-steps", default=60000, type=int, help="Number of parameters update, default=70000")
    parser.add_argument(
        "--num-steps-per-epoch", default=500, type=int, help="Number of steps to complete an epoch, default=500"
    )
    parser.add_argument(
        "--batch-size", default=2, type=int, help="Size of the batches used to update parameters, default=2"
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        default=[0.0001, 0.00001],
        type=float,
        help="List of learning rate values, default=[0.001, 0.0001]",
    )
    parser.add_argument(
        "--decay-steps",
        nargs="*",
        default=[50000],
        type=int,
        help="List of steps at which we decay the learning rate, default=[40000]",
    )
    return parser.parse_args()


def main():
    num_classes = len(class_names)
    image_shape = (600, 1987, 3)

    args = parse_args()

    pipeline_creator = InputPipelineCreator(num_classes=num_classes, image_shape=image_shape)
    dataset_train = pipeline_creator.create_input_pipeline(
        filename=args.train_data_path, batch_size=args.batch_size, training=True
    )
    dataset_valid = pipeline_creator.create_input_pipeline(args.valid_data_path)

    train_classification_loss = tf.keras.metrics.Mean(name="train_classification_loss")
    train_regression_loss = tf.keras.metrics.Mean(name="train_regression_loss")
    train_map_50 = metric_utils.MeanAveragePrecision(num_classes, 0.5, name="train_mAP@IoU=.50")
    train_map_75 = metric_utils.MeanAveragePrecision(num_classes, 0.75, name="train_mAP@IoU=.75")

    valid_classification_loss = tf.keras.metrics.Mean(name="valid_classification_loss")
    valid_regression_loss = tf.keras.metrics.Mean(name="valid_regression_loss")
    valid_map_50 = metric_utils.MeanAveragePrecision(num_classes, 0.5, name="valid_mAP@IoU=.50")
    valid_map_75 = metric_utils.MeanAveragePrecision(num_classes, 0.75, name="valid_mAP@IoU=.75")

    train_log_dir = os.path.join(args.logs_dir, "faster-rcnn", "train")
    valid_log_dir = os.path.join(args.logs_dir, "faster-rcnn", "valid")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=args.decay_steps, values=args.learning_rates
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model = FasterRCNN(image_shape, num_classes)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=os.path.join(args.checkpoints_dir, "faster-rcnn"), max_to_keep=1
    )

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        tf.print("Restored from {}".format(manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")

    for images, gt_classes, gt_boxes in dataset_train:
        checkpoint.step.assign_add(1)
        step = int(checkpoint.step)

        (cls_loss, reg_loss, pred_boxes, pred_scores, pred_classes) = model.train_step(
            images, gt_classes, gt_boxes, optimizer
        )

        train_classification_loss(cls_loss)
        train_regression_loss(reg_loss)
        train_map_50(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)
        train_map_75(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)

        if step % args.num_steps_per_epoch == 0:
            epoch = step // args.num_steps_per_epoch

            with train_summary_writer.as_default():
                tf.summary.scalar("Losses/classification_loss", train_classification_loss.result(), step=step)
                tf.summary.scalar("Losses/regression_loss", train_regression_loss.result(), step=step)
                tf.summary.scalar("Metrics/mAP@IoU=.50", train_map_50.result(), step=step)
                tf.summary.scalar("Metrics/mAP@IoU=.75", train_map_75.result(), step=step)

            with valid_summary_writer.as_default():
                for test_step, (images, gt_classes, gt_boxes) in dataset_valid.enumerate():

                    (cls_loss, reg_loss, pred_boxes, pred_scores, pred_classes) = model.test_step(
                        images, gt_classes, gt_boxes
                    )

                    valid_classification_loss(cls_loss)
                    valid_regression_loss(reg_loss)
                    valid_map_50(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)
                    valid_map_75(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)

                    if test_step == 0:
                        # Add ground-truth boxes
                        if epoch == 1:
                            image_gt_boxes = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
                            image_utils.draw_predictions_on_image(
                                image_gt_boxes,
                                gt_boxes[0],
                                class_indices=tf.math.argmax(gt_classes[0, :, 1:], -1),
                                class_names=class_names,
                                relative=True,
                            )
                            tf.summary.image("Ground-truth", image_utils.to_tensor(image_gt_boxes), step=0)
                            image_gt_boxes.close()

                        # Add predictions to summary every 5 epochs
                        if epoch % 1 == 0:
                            image_predictions_50 = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
                            pred_boxes_50 = tf.gather_nd(pred_boxes, tf.where(pred_scores > 0.5))
                            pred_scores_50 = tf.gather_nd(pred_scores, tf.where(pred_scores > 0.5))
                            pred_classes_50 = tf.gather_nd(pred_scores, tf.where(pred_scores > 0.5))
                            image_utils.draw_predictions_on_image(
                                image_predictions_50,
                                pred_boxes_50,
                                scores=pred_scores_50,
                                class_indices=tf.cast(pred_classes_50, tf.int32),
                                class_names=class_names,
                                relative=True,
                            )
                            tf.summary.image(
                                "Predictions/pred@score=.50", image_utils.to_tensor(image_predictions_50), step=step
                            )
                            image_predictions_50.close()

                            image_predictions_75 = Image.fromarray(tf.cast(images[0], dtype=tf.uint8).numpy())
                            pred_boxes_75 = tf.gather_nd(pred_boxes, tf.where(pred_scores > 0.75))
                            pred_scores_75 = tf.gather_nd(pred_scores, tf.where(pred_scores > 0.75))
                            pred_classes_75 = tf.gather_nd(pred_scores, tf.where(pred_scores > 0.75))
                            image_utils.draw_predictions_on_image(
                                image_predictions_75,
                                pred_boxes_75,
                                scores=pred_scores_75,
                                class_indices=tf.cast(pred_classes_75, tf.int32),
                                class_names=class_names,
                                relative=True,
                            )
                            tf.summary.image(
                                "Predictions/pred@score=.75", image_utils.to_tensor(image_predictions_75), step=step
                            )
                            image_predictions_75.close()

                tf.summary.scalar("Losses/classification_loss", valid_classification_loss.result(), step=step)
                tf.summary.scalar("Losses/regression_loss", valid_regression_loss.result(), step=step)
                tf.summary.scalar("Metrics/mAP@IoU=.50", valid_map_50.result(), step=step)
                tf.summary.scalar("Metrics/mAP@IoU=.75", valid_map_75.result(), step=step)

            # Print metrics of the epoch
            template = "Epoch {}/{}: \n"
            template += "\tCls Loss       --> Train: {:.2f}, Valid: {:.2f}\n"
            template += "\tReg Loss       --> Train: {:.2f}, Valid: {:.2f}\n"
            template += "\tmAP at IoU=.50 --> Train: {:.2f}, Valid: {:.2f}\n"
            template += "\tmAP at IoU=.75 --> Train: {:.2f}, Valid: {:.2f}\n"

            tf.print(
                template.format(
                    epoch,
                    args.num_steps // args.num_steps_per_epoch,
                    train_classification_loss.result(),
                    valid_classification_loss.result(),
                    train_regression_loss.result(),
                    valid_regression_loss.result(),
                    train_map_50.result(),
                    valid_map_50.result(),
                    train_map_75.result(),
                    valid_map_75.result(),
                )
            )

            # Reset metric objects
            train_classification_loss.reset_states()
            train_regression_loss.reset_states()
            train_map_50.reset_states()
            train_map_75.reset_states()
            valid_classification_loss.reset_states()
            valid_regression_loss.reset_states()
            valid_map_50.reset_states()
            valid_map_75.reset_states()

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                save_path = manager.save()
                tf.print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        # Save model's weights when training is finished
        if step == args.num_steps:
            model.save_weights(os.path.join(args.save_dir, "faster-rcnn", "weights"), save_format="tf")
            break


if __name__ == "__main__":
    main()