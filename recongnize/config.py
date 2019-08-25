
class Config(dict):
    def __setattr__(self, key, value):
        self[key]=value
    def __getattr__(self, item):
        try:
            v=self[item]
            return v
        except:
            raise Exception('Config object has no key %s '%item)
    def add(self,key,value,description=''):
        self[key]=value
num_classes = 31 + 26 + 10

maxPrintLen = 100
charset = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"
           ]

ROOT='recongnize'
config=Config()

config.add('restore', True, 'whether to restore from the latest checkpoint')
config.add('checkpoint_dir','data/checkpoint/', 'the checkpoint dir')
config.add('initial_learning_rate', 1e-3, 'inital lr')

config.add('image_height', 72, 'image height')
config.add('image_width', 272, 'image width')
config.add('image_channel', 3, 'image channels as input')



config.add('cnn_count', 4, 'count of cnn module to extract image features.')
config.add('out_channels', 64, 'train channels of last layer in CNN')
config.add('num_hidden', 128, 'number of hidden units in lstm')
config.add('output_keep_prob', 0.8, 'output_keep_prob in lstm')

config.add('leakiness', 0.01, 'leakiness of lrelu')
config.add('decay_rate', 0.98, 'the lr decay rate')
config.add('beta1', 0.9, 'parameter of adam optimizer beta1')
config.add('beta2', 0.999, 'adam parameter beta2')

config.add('decay_steps', 10000, 'the lr decay_step for optimizer')
config.add('momentum', 0.9, 'the momentum')


config.add('num_epochs', 10000, 'maximum epochs')
config.add('batch_size', 40, 'the batch_size')
config.add('save_steps', 500, 'the step to save checkpoint')
config.add('validation_steps', 500, 'the step to validation')


config.add('train_dir', './imgs/train_2/', 'the train data dir')
config.add('val_dir', './imgs/val/', 'the val data dir')
config.add('infer_dir', './imgs/infer/', 'the infer data dir')
config.add('tf_log_dir', './data/log', 'the logging dir')
config.add('num_gpus', 1, 'num of gpus')

FLAGS=config
