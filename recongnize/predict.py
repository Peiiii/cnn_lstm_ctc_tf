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
from .utils  import tools
from . import config as cfg


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = cfg.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)



class Recongizer:
    def __init__(self):
        self.imread_mode = cv2.IMREAD_COLOR
        self.img_size=(272,72)

        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        with self.graph.as_default():
            self.model = cnn_lstm_otc_ocr.LSTMOCR('infer',batch_size=1)
            self.model.build_graph()
            self.saver=tf.train.Saver()
            # 注意！恢复器必须要在新创建的图里面生成,否则会出错。
        self.sess = tf.Session(graph=self.graph)  # 创建新的sess


        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                print('loading model  %s ...'%(ckpt))
                self.saver.restore(self.sess, ckpt)  # 从恢

    def predict_from_file(self,fp):
        img=cv2.imread(fp)
        img = (cv2.resize(img, self.img_size) / 255) * 2 - 1
        return self.predict(img)

    def predict(self,img):
        # print(img)
        # print(img.shape)
        img=(cv2.resize(img,self.img_size)/255)*2-1
        img=np.array([img])
        y=self._predict(img)[0]
        return y

    def _predict(self,xs):
        feed = {self.model.inputs: xs}
        ys = self.sess.run(self.model.dense_decoded, feed)
        ys=self.decodePreds(ys)
        return ys

    def decodePreds(self,ys, verbose=False):
        letters2 = cfg.charset

        def tensor_to_text(y):
            text = [letters2[i] for i in y]
            text = ''.join(text)
            return text

        if verbose:
            print('indexes:', ys)
        ys = [tensor_to_text(y) for y in ys]
        return ys




if __name__=="__main__":
    # xs,labels,file_names=loadXY('imgs/infer')
    # file_names=glob.glob('/home/user/datasets/results/ccpd/ccpd_chanllenge_results/*/*.jpg')
    # predict_test()
    # predict_dir('demo_imgs')
    pass