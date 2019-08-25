from recongnize.train import Trainer
from recongnize.predict import Recongizer

def train():
    T=Trainer()
    T.train()
def test():
    R=Recongizer()

    y=R.predict_from_file('data/demo/6.jpg')
    print(y)

if __name__=="__main__":

    # demo()
    train()
    # test()