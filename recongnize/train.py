"""

"""

import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from . import cnn_lstm_otc_ocr
from .utils import tools as utils
from .utils import data_iterator as iterator
import os
from . import config as cfg

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = cfg.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


class Trainer:
    def __init__(self):

        pass


    def train(self,train_dir='/home/user/datasets/plate/ccpd_detect_results_train',  val_dir='/home/user/datasets/plate/ccpd_detect_results_val', mode='train'):

        model = cnn_lstm_otc_ocr.LSTMOCR(mode)
        model.build_graph()

        # train_feeder = iterator.DataGenerator(batch_size=FLAGS.batch_size)
        train_feeder = iterator.DataLoader(data_dir=train_dir, batch_size=FLAGS.batch_size)
        print('loading validation data')
        val_feeder = iterator.DataLoader(data_dir=val_dir, batch_size=FLAGS.batch_size)
        print('size: {}\n'.format(val_feeder.num_files))

        num_batches_per_epoch = 1000
        num_batches_val = val_feeder.num_files // val_feeder.batch_size

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(FLAGS.tf_log_dir + '/train', sess.graph)

            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                train_cost_all = []
                train_acc_all = []

                # the training part
                for cur_batch in range(num_batches_per_epoch):

                    batch_inputs, (_, batch_labels_plain, batch_labels) = train_feeder.getBatch()
                    # print(len(batch_inputs))
                    # print(batch_labels[-1])
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels}

                    # if summary is needed

                    summary_str, preds_decoded, batch_cost, step, _ = \
                        sess.run(
                            [model.merged_summay, model.dense_decoded, model.cost, model.global_step, model.train_op],
                            feed)

                    batch_acc = utils.accuracy_calculation(batch_labels_plain, preds_decoded)

                    # print('train** origin: %s ,prediction: %s'%(batch_labels_plain[0],preds_decoded[0]))

                    train_cost_all.append(batch_cost)
                    train_acc_all.append(batch_acc)

                    print('step:  %s' % step)
                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save checkpoint at step {0}', format(step))
                        print('save checkpoint at step %s' % (step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)

                    # do validation
                    if step % FLAGS.validation_steps == 0:
                        val_acc_all = []
                        val_cost_all = []
                        lr = 0

                        for j in range(num_batches_val):
                            val_inputs, (_, ori_labels, val_labels) = val_feeder.getBatch()
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels}
                            dense_decoded, val_cost_batch, lr = \
                                sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                                         val_feed)
                            val_acc_batch = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                                       ignore_value=-1, isPrint=True)
                            # print('val** origin: %s ,prediction: %s' % (ori_labels[0], dense_decoded[0]))

                            val_acc_all.append(val_acc_batch)
                            val_cost_all.append(val_cost_batch)

                        val_acc = np.average(val_acc_all)
                        val_cost = np.average(val_cost_all)
                        train_acc = np.average(train_acc_all)
                        train_cost = np.average(train_cost_all)

                        print('train_acc: %.4f , train_cost: %.4f' % (train_acc, train_cost),
                              'val_acc: %.4f , val_cost: %.4f' % (val_acc, val_cost))

                train_acc = np.average(train_acc_all)
                train_cost = np.average(train_cost_all)

                print('train_acc: %.4f , train_cost: %.4f' % (train_acc, train_cost))










if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    Trainer().train('/home/user/datasets/plate/ccpd_detect_results_train', '/home/user/datasets/plate/ccpd_detect_results_val', FLAGS.mode)

