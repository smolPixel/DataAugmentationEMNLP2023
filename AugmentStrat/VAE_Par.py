"""VAE augmentation"""
import pandas
from AugmentStrat.VAE_strat.VAE import VAE_meta
from data.DataProcessor import separate_per_class
import torch
from AugmentStrat.VAE_strat.utils import to_var
import math
import itertools
import numpy as np
import random
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
class VAE_Par():

    def __init__(self, argdict):
        self.argdict=argdict
        self.algo_is_trained=False


    #Next: train batch pos, batch neg
    def init_model(self, training_set):
        self.vae=VAE_meta(self.argdict, training_set)
                # if self.argdict['tie_embedding']:

        # fds

    def train_model(self):

        for i in range(self.argdict['nb_epoch_algo']):
            NLL = []
            KL = []
            KL_weight = []
            data_loader = DataLoader(
                dataset=self.vae.train,
                batch_size=32,  # self.argdict.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            # tracker = defaultdict(tensor)

            # data_loader_p=iter(data_loader_pos)
            iteration=0
            for i, batch in enumerate(data_loader):
                nll_batch, KL_batch, weight_batch= self.vae.run_batches(batch, iteration, data_loader)
                iteration+=1
                NLL.append(nll_batch)
                KL_weight.append(weight_batch)
                KL.append(KL_batch)
            # print(f"Epoch {i} KL div {np.mean(KL_neg)}/{np.mean(KL_pos)} KL Weight {np.mean(KL_weight_neg)}/{np.mean(KL_weight_pos)}, "
            #       f"NLL {np.mean(NLL_neg)}/{np.mean(NLL_pos)}")
        # for vae in self.vae_per_class:
        #     print(vae.generate())
        # fds


    def augment(self, training_set, dev_set):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(training_set)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        num_per_cat = round(num_points * split) / len(self.argdict['categories'])
        diconew = {}
        bs=50
        if not self.algo_is_trained:
            self.init_model(training_set)
            self.train_model()

        data_loader = DataLoader(
            dataset=training_set,
            batch_size=32,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )


        for batch in data_loader:

            bs=batch['input'].shape[0]
            encoded=self.vae.encode_examples(batch['input'].cuda())
            samples, z = self.vae.model.inference(n=bs, z=encoded)
            generated = training_set.arr_to_sentences(samples)
            for i, sentAug in enumerate(generated):
                if sentAug=="":
                    print("WARNING ERRONEOUS SENTENCE")
                    sentAug="Error Temp"
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':batch['label'][i].item(),
                                 'input':training_set.tokenize(sentAug),
                                 'augmented':True}
        self.algo_is_trained = True
        for j, item in diconew.items():
            len_data=len(training_set)
            # print(item)
            training_set.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        # training_set.return_pandas().to_csv(f'VAEPar_train_{self.argdict["dataset"]}.tsv', sep='\t')
        # fds
        return training_set

        # Creating new dataset
