import numpy as np
import os,random,cv2
from ..config import charset
from .genPlate.genPlate import G

def genBatch(batch_size,size ):
    return G.genBatch_4(batch_size,size=size)

def text_to_tensor(text):
    idxes=[charset.index(char) for char in text]
    idxes=np.array(idxes)
    return idxes
def tensor_to_text(tensor):
    text=[charset[i] for i in tensor]
    text=''.join(text)
    return text

class DataLoader:
    def __init__(self,data_dir,batch_size=128):
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.imread_mode=cv2.IMREAD_COLOR
        self.build()
    def build(self):
        file_names=os.listdir(self.data_dir)
        self.file_names=[self.data_dir+'/'+f for f in file_names]
        self.num_files=len(self.file_names)
        self.num_batches=self.num_files//self.batch_size
        self.current_batch_index=-1
        random.shuffle(self.file_names)
    def getBatch(self):
        self.current_batch_index+=1
        if self.current_batch_index==self.num_batches:
            random.shuffle(self.file_names)
            self.current_batch_index=0
        batch=self.getBatchByIndex(self.current_batch_index)
        return batch
    def getBatchByIndex(self,batch_index):
        file_names=self.file_names[batch_index*self.batch_size : (batch_index+1)*self.batch_size]
        batch=self.loadData(file_names)
        return batch
    def loadData(self,file_names):
        X_data=[]
        Y_data=[]
        for f in file_names:
            img=cv2.imread(f,self.imread_mode)
            img=cv2.resize(img,(272,72))
            X_data.append(img)
            label=self.getLabel(f)
            Y_data.append(label)
        X_data=self.preprocess_X(X_data)
        Y_data=self.encode_labels(Y_data)
        return X_data,Y_data
    def preprocess_X(self,xs):
        xs = np.array(xs)
        xs=(xs/255)*2-1
        return xs
    def getLabel(self,f):
        f=os.path.basename(f)
        label=f.split('_')[-1][:7]
        return label
    def encode_labels(self,labels):
        ys_text=labels
        ys=[self.encode_a_label(label) for label in labels]
        ys_encoded=np.array(ys)
        ys_encoded_sparse=sparse_tuple_from_label(ys)
        return ys_text,ys_encoded,ys_encoded_sparse
    def encode_a_label(self,y):
        return text_to_tensor(y)

class DataGenerator:

    def __init__(self,batch_size=128):
        self.batch_size=batch_size
        self.current_batch={}
        self.img_size=(272,72)

    def getBatch(self,batch_size=None):
        if not  batch_size:
            batch_size=self.batch_size

        X_data,Y_data=genBatch(batch_size,self.img_size)
        X_data=self.preprocess_X(X_data)
        Y_data=self.encode_labels(Y_data)
        self.current_batch={'x':X_data,'y':Y_data}
        return X_data,Y_data

    def preprocess_X(self, xs):
        xs = np.array(xs)
        xs = (xs / 255) * 2 - 1
        return xs

    def encode_labels(self, labels):
        ys_text = labels
        ys = [self.encode_a_label(label) for label in labels]
        ys_encoded = np.array(ys)
        ys_encoded_sparse = sparse_tuple_from_label(ys)
        return ys_text, ys_encoded, ys_encoded_sparse
    def encode_a_label(self,y):
        return text_to_tensor(y)




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

