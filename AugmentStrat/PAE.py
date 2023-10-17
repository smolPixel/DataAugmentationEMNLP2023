"""VAE augmentation"""
import pandas
from AugmentStrat.PAE_strat.PAE import PAE_meta
from data.DataProcessor import separate_per_class
import torch
from AugmentStrat.PAE_strat.utils import to_var
import math
import itertools
import numpy as np
import random
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
class PAE():

    def __init__(self, argdict):
        self.argdict=argdict
        self.algo_is_trained=False
        self.vae_per_class=[None]*len(self.argdict['categories'])


    def sample_latent_space(self, bs, encoded):
        if self.argdict['sampling_strategy']=="random":
            return to_var(torch.randn([bs, self.argdict['latent_size']]))
        elif self.argdict['sampling_strategy']=="grid":
            root=math.ceil(bs**(1/self.argdict['latent_size']))
            #Spans from -1 to 1, we need root spaces
            largeur_col=3/float((root-1))

            dim = [-1.5 + largeur_col * i for i in range(root)]
            all_possibilities = []
            for comb in itertools.product(dim, repeat=self.argdict['latent_size']):
                all_possibilities.append(comb)
            point=torch.zeros(bs, self.argdict['latent_size'])
            points_chosen=np.random.choice(np.arange(len(all_possibilities)), size=bs, replace=False)
            points_latent=torch.zeros([bs, self.argdict['latent_size']])
            for i, pp in enumerate(points_chosen):
                comb=all_possibilities[pp]
                points_latent[i]=torch.Tensor(list(comb))
            return points_latent
        elif self.argdict['sampling_strategy']=='posterior':
            enco=encoded['encoded']
            points_latent = torch.zeros([bs, self.argdict['latent_size']])
            num_points=encoded['encoded'].shape[0]
            for i in range(bs):
                random_zero=random.randint(0, num_points-1)
                random_one=random.randint(0, num_points-1)
                interpol=random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                points_latent[i]=enco[random_zero]*interpol+enco[random_one]*(1-interpol)
            # print(points_latent)
            # pass
            return points_latent

    #Next: train batch pos, batch neg
    def init_models(self, training_sets):
        for classe, training_set in enumerate(training_sets):
            vae=PAE_meta(self.argdict, training_set)
            if classe != 0 and self.argdict['tie_embeddings']:
                raise ValueError("ff")
                vae.model.embedding=self.vae_per_class[0].model.embedding
            if classe !=0 and self.argdict['tie_encoders']:
                raise ValueError("ff")
                vae.model.encoder_rnn=self.vae_per_class[0].model.encoder_rnn
                vae.model.hidden2mean=self.vae_per_class[0].model.hidden2mean
                vae.model.hidden2logv=self.vae_per_class[0].model.hidden2logv
            self.vae_per_class[classe]=vae
                # if self.argdict['tie_embedding']:

        # fds

    def train_models(self):
        # for vaes in self.vae_per_class:




        # for i in range(self.argdict['nb_epoch_algo']):
        num_cat=len(self.argdict['categories'])
        NLL = [[] for i in range(num_cat)]
        KL = [[] for i in range(num_cat)]
        KL_weight = [[] for i in range(num_cat)]
        data_loaders=[DataLoader(
            dataset=self.vae_per_class[i].train,
            batch_size=4,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        ) for i in range(num_cat)]

        # data_loader_neg = DataLoader(
        #     dataset=self.vae_per_class[0].train,
        #     batch_size=32,  # self.argdict.batch_size,
        #     shuffle=True,
        #     num_workers=cpu_count(),
        #     pin_memory=torch.cuda.is_available()
        # )

        # tracker = defaultdict(tensor)
        print(data_loaders)
        # data_loaders=[iter(data_loader) for data_loader in data_loaders]
        # data_loader_neg=iter(data_loader_neg)
        iteration=0
        #Unfortunately everything is shit so I have to do it like this for large classes
        for cc in range(num_cat):
            print(f"Training VAE of class {cc}")
            iteration=0
            NLL = []
            KL = []
            KL_weight = []
            for i in range(self.argdict['nb_epoch_algo']):

                dl=DataLoader(
                    dataset=self.vae_per_class[cc].train,
                    batch_size=32,  # self.argdict.batch_size,
                    shuffle=True,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available())

                for batch in dl:
                    nll_batch, KL_batch, weight_batch= self.vae_per_class[cc].run_batches(batch, iteration, data_loaders[cc])
                    self.vae_per_class[cc].interpolate(5)
                    fds
                    iteration+=1
                    NLL.append(nll_batch)
                    KL_weight.append(weight_batch)
                    KL.append(KL_batch)
                # fds
                # nll_batch_neg, KL_batch_neg, weight_batch_neg= self.vae_per_class[0].run_batches(batch_neg, iteration, data_loader_neg)
                # nll_batch_pos, KL_batch_pos, weight_batch_pos =self.vae_per_class[1].run_batches(batch_pos, iteration, data_loader_pos)
                # iteration+=1
                # NLL_neg.append(nll_batch_neg)
                # NLL_pos.append(nll_batch_pos)
                # KL_weight_neg.append(weight_batch_neg)
                # KL_weight_pos.append(weight_batch_pos)
                # KL_neg.append(KL_batch_neg)
                # KL_pos.append(KL_batch_pos)
                # print(f"Epoch {i} KL div {np.mean(KL)} KL Weight {np.mean(KL_weight)}, "
                #       f"NLL {np.mean(NLL)}")
        # for vae in self.vae_per_class:
        #     print(vae.generate())
        # fds


    def augment(self, train, dev):
        training_sets=separate_per_class(train)
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        num_per_cat = round(num_points * split) / len(training_sets)
        diconew = {}
        bs=50

        if not self.algo_is_trained:
            self.init_models(training_sets)
            self.train_models()

        for classe, training_set in enumerate(training_sets):
            # if not self.algo_is_trained:
            #     vae=VAE_meta(self.argdict, training_set)
            #     vae.train_model()
            #     self.vae_per_class[classe]=vae
            #     # if self.argdict['tie_embedding']:
            #
            # else:
            vae=self.vae_per_class[classe]

            encoded=vae.encode()
            all_points=self.sample_latent_space(int(num_per_cat), encoded)
            num_generated=0
            start=0
            # print("BITCH")
            while num_generated<int(num_per_cat):
                # points = self.sample_latent_space(bs)
                points = self.sample_latent_space(int(bs), encoded)
                # points = all_points[start:start+bs]
                # start=start+bs
                # print(points)
                # fds

                points = points.cuda()
                # print(points)
                samples, z = vae.model.inference(n=bs, z=points)
                generated = train.arr_to_sentences(samples)
                for sentAug in generated:
                    if sentAug=="":
                        print("WARNING ERRONEOUS SENTENCE")
                        sentAug="Error Temp"
                    diconew[len(diconew)]={'sentence':sentAug,
                                     'label':classe,
                                     'input':train.tokenize(sentAug),
                                     'augmented':True}
                    # print(sentAug)
                    num_generated+=1
                    if num_generated==num_per_cat:
                        break
        
        self.algo_is_trained = True
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train

        # Creating new dataset
