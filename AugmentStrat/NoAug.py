"""EDA augmentation"""
import pandas

class NoAug():

    def __init__(self, argdict):
        self.argdict=argdict


    def augment(self, train, return_dict=False):
        # print(train)
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train
