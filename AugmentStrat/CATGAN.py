# import AugmentStrat.CATGAN_strat.config as cfg
# from AugmentStrat.CATGAN_strat.utils.text_process import load_test_dict, text_process
from AugmentStrat.CATGAN_strat.catgan_instructor import CatGANInstructor
from AugmentStrat.CATGAN_strat.utils.create_opt import create_opt
from AugmentStrat.CATGAN_strat.utils.text_process import text_process, load_test_dict
import pandas as pd
import os
import torch


class CatGan():

    def __init__(self, argdict):
        self.argdict = argdict
        self.opt=create_opt()
        self.instructor=None#CatGANInstructor(self.opt)
        ds="AugmentStrat/CATGAN_strat/dataset"
        self.algo_is_trained = False
        # for file in os.listdir(ds):
        #     # print(file)
        #     os.remove(ds+'/'+file)

        # asdf

    def augment(self, train, return_dict=False):
        dev=pd.read_csv(f'data/{self.argdict["dataset"]}/dev.tsv', sep='\t')
        # print(dev)
        # print(train.return_pandas())
        if not self.algo_is_trained:
            self.dump_data(train.return_pandas(), dev)
            # dsa
            self.opt.max_seq_len, self.opt.vocab_size = text_process('AugmentStrat/CATGAN_strat/dataset/' + self.opt.dataset + '.txt')
            self.opt.extend_vocab_size = len(load_test_dict(self.opt.dataset, self.opt.path)[0])  # init classifier vocab_size
            self.instructor=CatGANInstructor(self.opt)
            self.instructor._run()
            self.algo_is_trained=True
        num_to_add=int((self.argdict['dataset_size'])*self.argdict['split']/len(self.argdict['categories']))

        new_data = {}

        for i in range(len(self.argdict['categories'])):
            with torch.no_grad():
                samples=self.instructor._sample(num_to_add, i)
            for j, sam in enumerate(samples):
                len_data = len(train)
                sentAug=" ".join(sam)

                new_data[len(new_data)] = {'sentence': sentAug,
                                           'label': i,
                                           'input': train.tokenize(sentAug),
                                           'augmented': True}
        print('--------------')
        print(torch.cuda.memory_allocated())
        if return_dict:
            return new_data
        for j, item in new_data.items():
            len_data = len(train)
            # print(item)
            train.data[len_data] = item

        print(torch.cuda.memory_allocated())
        return train

    def augment_doublons_algo(self, train, n):
        dev=pd.read_csv(f'data/{self.argdict["dataset"]}/dev.tsv', sep='\t')
        # print(dev)
        # print(train.return_pandas())
        self.dump_data(train.return_pandas(), dev)
        num_to_add = int((self.argdict['dataset_size']) * self.argdict['split'] / len(self.argdict['categories']))
        # dsa
        self.opt.max_seq_len, self.opt.vocab_size = text_process('AugmentStrat/CATGAN_strat/dataset/' + self.opt.dataset + '.txt')
        self.opt.extend_vocab_size = len(load_test_dict(self.opt.dataset, self.opt.path)[0])  # init classifier vocab_size
        self.instructor=CatGANInstructor(self.opt)
        self.instructor._run()
        numFalsePos = 0
        numFalseNeg = 0
        diconew = {}
        j=0
        for i in range(len(self.argdict['categories'])):

            print(num_to_add)
            samples=self.instructor._sample(num_to_add*10, i)
            doublons, not_doublons=self.extract_doublons(samples, train)
            assert len(doublons)>n
            assert len(not_doublons)>(num_to_add-n)
            if n==0:
                sent_to_add=not_doublons[:num_to_add]
            else:
                print(n)
                print(num_to_add-n)
                doublons[:n]
                not_doublons[:(num_to_add-n)]
                sent_to_add=doublons[:int(n)]+not_doublons[:int(num_to_add-n)]
            for sentAug in sent_to_add:
                diconew[j] = {'sentence': sentAug,
                              'label': i,
                              'input': train.tokenize(sentAug)}
                j+=1

        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train[len_data]=item
        return train

    def extract_doublons(self, samples, train):
        # print(samples)
        trainSent=train.return_pandas()
        trainSent.to_csv('temp.csv')
        trainSent=list(trainSent['sentence'])
        doublons=[]
        not_doublons=[]
        for sam in samples:
            sam=" ".join(sam)
            if sam in trainSent:
                doublons.append(sam)
            else:
                not_doublons.append(sam)
        return doublons, not_doublons

    def dump_data(self, train, dev):
        """In: dfs, out: dumps the data for CATGAN"""
        #train
        sentences=train['sentence']
        labels=train['label']
        all_sentences=""
        sent_by_class=[""]*len(self.argdict['categories'])
        for sent, lab in zip(sentences, labels):
            all_sentences+=sent+"\n"
            sent_by_class[lab]+=sent+"\n"

        with open(f"AugmentStrat/CATGAN_strat/dataset/{self.argdict['dataset']}.txt", "w") as f:
            f.write(all_sentences)

        for cat in range(len(sent_by_class)):
            with open(f"AugmentStrat/CATGAN_strat/dataset/{self.argdict['dataset']}_cat{cat}.txt", "w") as f:
                f.write(sent_by_class[cat])

        # dev
        sentences = dev['sentence']
        labels = dev['label']
        all_sentences = ""
        sent_by_class = [""] * len(self.argdict['categories'])
        for sent, lab in zip(sentences, labels):
            all_sentences += sent + "\n"
            sent_by_class[lab] += sent + "\n"



        with open(f"AugmentStrat/CATGAN_strat/dataset/{self.argdict['dataset']}_test.txt", "w") as f:
            f.write(all_sentences)

        for cat in range(len(sent_by_class)):
            with open(f"AugmentStrat/CATGAN_strat/dataset/{self.argdict['dataset']}_cat{cat}_test.txt", "w") as f:
                f.write(sent_by_class[cat])


