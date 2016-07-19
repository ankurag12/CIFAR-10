from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import read_data
import model_cnn
import tensorflow as tf

BATCH_SIZE = 128
NUM_EPOCHS = 100
INITIAL_LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 5
NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES

def run_training():
    with tf.Graph().as_default():
        train_images, train_labels = read_data.inputs(data_set='train', batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        # Don't specify number of epochs in validation set, otherwise that limits the training duration as the
        # validation set is 10 times smaller than the training set
        val_images, val_labels = read_data.inputs(data_set='validation', batch_size=BATCH_SIZE, num_epochs=None)

        with tf.variable_scope("trained_variables"):    # THIS IS VERY IMPORTANT
            train_logits = model_cnn.inference(train_images)
            train_accuracy = model_cnn.evaluation(train_logits, train_labels)
            tf.get_variable_scope().reuse_variables()   # THIS IS VERY IMPORTANT
            val_logits = model_cnn.inference(val_images)
            val_accuracy = model_cnn.evaluation(val_logits, val_labels)

        loss = model_cnn.loss(train_logits, train_labels)

        decay_steps = int(EPOCHS_PER_LR_DECAY * NUM_TRAIN_EXAMPLES / BATCH_SIZE)
        train_op = model_cnn.training(loss, INITIAL_LEARNING_RATE, decay_steps, LR_DECAY_FACTOR)

        init_op = tf.initialize_all_variables()

        sess = tf.Session()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                _, loss_value, train_acc_val, valid_acc_val = sess.run([train_op, loss, train_accuracy, val_accuracy])

                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    print('Step %d : loss = %.5f , training accuracy = %.1f, validation accuracy = %.1f (%.3f sec)'
                          % (step, loss_value, train_acc_val, valid_acc_val, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps' % (NUM_EPOCHS, step))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()