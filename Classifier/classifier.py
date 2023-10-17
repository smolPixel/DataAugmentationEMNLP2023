"""Labellers remplace the oracle in this version"""
import torch
import pandas as pd
# from Labellers.labeller_template import labeller_template

class classifier():

    def __init__(self, argdict, train):
        if argdict['classifier']=='dummy':
            from Classifier.dummy import dummy_Classifier
            self.algo=dummy_Classifier(argdict)
        elif argdict['classifier']=='LogReg':
            from Classifiers.LogReg import LogReg_Classifier
            self.algo=LogReg_Classifier(argdict)
        elif argdict['classifier']=='Linear':
            from Classifier.LinearLayer import Linear_Classifier
            self.algo=Linear_Classifier(argdict, train)
        elif argdict['classifier']=='rnn':
            from Classifier.RNN import RNN_Classifier
            self.algo=RNN_Classifier(argdict, train)
        elif argdict['classifier'].lower()=="bert":
            from Classifier.BERT import Bert_Classifier
            self.algo=Bert_Classifier(argdict)
        elif argdict['classifier'].lower()=="bertaeda":
            from Classifier.BERTAEDA import Bert_AEDA_Classifier
            self.algo=Bert_AEDA_Classifier(argdict)
        elif argdict['classifier'].lower()=="mbert":
            from Classifier.mBERT import mBert_Classifier
            self.algo=mBert_Classifier(argdict)
        elif argdict['classifier'].lower()=="mbertaeda":
            from Classifier.mBERTAEDA import mBert_AEDA_Classifier
            self.algo=mBert_AEDA_Classifier(argdict)
        elif argdict['classifier']=='bert_autoSelect':
            from Classifier.BERT_AutoSelect import Bert_AutoSelect_Classifier
            self.algo=Bert_AutoSelect_Classifier(argdict)
        elif argdict['classifier']=='jiant':
            from Classifier.jiant_classifier import Jiant_Classifier
            self.algo=Jiant_Classifier(argdict)
        elif argdict['classifier']=='svm_latent':
            from Classifiers.svm_latent import SVM_Latent_Classifier
            self.algo=SVM_Latent_Classifier(argdict)
        else:
            raise ValueError(f"No classifier named {argdict['classifier']}")
        self.argdict=argdict
        # self.trainData, self.devData= self.load_data()

    def init_model(self):
        self.algo.init_model()

    def save_state(self, path):
        torch.save(self.algo.state_dict(), path)

    def train_test(self, datasetTrain, datasetDev, datasetTest, generator=None, return_grad=False):
        """Receive as argument a dataloader from pytorch"""
        return self.algo.train_model(datasetTrain, datasetDev, datasetTest, generator, return_grad=return_grad)
    # def load_data(self):
    #     dfTrain=pd.read_csv(f'{self.argdict["pathFolder"]}/Dataset/SST-2/train.tsv', sep='\t')
    #     dfDev=pd.read_csv(f'{self.argdict["pathFolder"]}/Dataset/SST-2/dev.tsv', sep='\t')
    #     return dfTrain, dfDev
    #
    # def train(self):
    #     """Train the labeller"""
    #     self.algo.train(self.trainData, self.devData)
    #
    # def label(self, sentences):
    #     return self.algo.label(sentences)

    def train_test_batches(self,datasetTrain, datasetDev, generator=None, return_grad=False):
        """Train and test on individual batches to create dataset"""
        return self.algo.train_batches(datasetTrain, datasetDev)


    def separate_good_bad(self, datasetTrain, datasetDev):
        return self.algo.separate_good_bad(datasetTrain, datasetDev)

    def encode(self, dataset, generator=None):
        """Encode a dataset and return a tuple of (predicted_label, certitude)"""
        return self.algo.predict(dataset, generator)

    def label(self, sentences):
        labels, confidence = self.algo.label(sentences)
        return labels, confidence

    def get_logits(self, sentences):
        #get directly the logits
        logits=self.algo.get_logits(sentences)
        return logits

    def get_loss(self, sentences):
        return self.algo.get_loss(sentences)

    def get_grads(self, sentences, labels):
        grads=self.algo.get_grads(sentences, labels)
        return grads

    def get_rep(self, sentences):
        #get reprensentation of sentences
        return self.algo.get_rep(sentences)