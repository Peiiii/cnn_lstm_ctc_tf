"""

"""

import os,glob
import numpy as np
import tensorflow as tf
import cv2
from .genPlate.genPlate import G
from  .. import config as cfg
# +-* + () + 10 digit + blank + space
# num_classes = 31 + 26 + 10
#
maxPrintLen = 100
FLAGS=cfg.FLAGS
charset = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"
           ]
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


letters=charset
def labels_to_text(labels):  # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):  # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


def get_input_lens(sequences):
    # 64 is the train channels of the last layer of CNN
    lengths = np.asarray([FLAGS.out_channels for _ in sequences], dtype=np.int64)

    return lengths
class DataGenerator:
    def __init__(self):
        self.batch_size=FLAGS.batch_size
        self.max_text_len=7
    def genBatch(self):
        X_data,Y_text_data=G.genBatch_4(self.batch_size,size=(272,72),verbose=0)
        X_data=(X_data/255)*2-1
        Y_data = np.ones([self.batch_size, self.max_text_len])
        for i in range(self.batch_size):
            Y_data[i]=text_to_labels(Y_text_data[i])
        labels=Y_data
        Y_data=sparse_tuple_from_label(Y_data)
        # print(Y_data)
        lens=get_input_lens(X_data)
        return X_data,lens,Y_data,labels

class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                im = cv2.imread(image_name, cv2.IMREAD_COLOR).astype(np.float32) / 255.
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                im = im * 2 - 1

                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the train channels of the last layer of CNN
            lengths = np.asarray([FLAGS.out_channels for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False,log_file='./data/test.sv'):
    if len(original_seq) != len(decoded_seq):
        print('original_seq: %s\ndecoded_seq: %s' % (len(original_seq), len(decoded_seq)))
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open(log_file, 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')
        # print('origin_labeles:',origin_label)
        # print('decoded_labels:',decoded_label)
        if list(origin_label) == list(decoded_label):
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs




class ListIterator:
    def __init__(self,mylist,batch_size):
        self.list=mylist
        self.batch_size=batch_size
        self.length=len(mylist)
        self.num_batch=self.length//self.batch_size
        self.cur_batch_idx=-1
        self.cur_batch=None
    def next(self):
        self.cur_batch_idx+=1
        if self.cur_batch_idx>=self.num_batch:
            self.cur_batch_idx=0
        self.cur_batch=self.get_batch_by_idx(self.cur_batch_idx)
        return self.cur_batch

    def get_batch_by_idx(self,batch_idx):
        st=batch_idx*self.batch_size
        ed=st+self.batch_size
        return self.list[st:ed]





def loadInput(glob_str,imread_mode=cv2.IMREAD_COLOR,target_size=(272,72)):
    # file_names=[input_dir+'/'+f for f in os.listdir(input_dir)]
    file_names=glob.glob(glob_str)
    print(file_names)
    imgs=[cv2.imread(f,imread_mode).astype(np.float32) for f in file_names]
    imgs=np.array([cv2.resize(img,target_size) for img in imgs])
    imgs=(imgs/255)*2-1
    return imgs,file_names

def loadXY(input_dir,imread_mode=cv2.IMREAD_COLOR,target_size=(272,72),with_file_names=True):
    def getLabel(fn):
        label=os.path.basename(fn).split('_')[-1][:7]
        return label
    file_names=[input_dir+'/'+f for f in os.listdir(input_dir)]
    labels=[getLabel(fn) for fn in file_names]
    imgs=[cv2.imread(f,imread_mode).astype(np.float32) for f in file_names]
    imgs=np.array([cv2.resize(img,target_size) for img in imgs])
    imgs=(imgs/255)*2-1
    return imgs,labels,file_names


if __name__=="__main__":
    # D=DataGenerator()
    # I=DataIterator('imgs/val')
    # x1,_,y1,_=D.genBatch()
    # x2,_,y2=I.input_index_generate_batch([1,2])
    # print('y1:',y1)
    # print('y2:',y2)
    pass

