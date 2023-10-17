"""VAE augmentation"""
import pandas
from AugmentStrat.CVAE_strat.CVAE import CVAE_meta
from data.DataProcessor import separate_per_class
import torch
from AugmentStrat.VAE_strat.utils import to_var
class CVAE():

    def __init__(self, argdict):
        self.argdict=argdict
        self.algo_is_trained=False
        self.cvae=None

    def augment(self, train, dev, return_dict=False):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        num_per_cat = round(num_points * split) / len(self.argdict['categories'])
        diconew = {}
        bs=50
        if not self.algo_is_trained:
            cvae=CVAE_meta(self.argdict, train)
            cvae.train_model()
            self.cvae=cvae
            self.algo_is_trained = True
        else:
            cvae=self.cvae


        for classe in range(len(self.argdict['categories'])):
            for i in range(int(num_per_cat/bs)):
                points = to_var(torch.randn([bs, self.argdict['latent_size']]))

                points = points.cuda()
                # print(points)
                samples, z = cvae.model.inference(n=bs, z=points, cat=classe)
                generated = train.arr_to_sentences(samples)
                for sentAug in generated:
                    diconew[len(diconew)]={'sentence':sentAug,
                                     'label':classe,
                                     'input':train.tokenize(sentAug),
                                     'augmented':True}
            # fds

        if return_dict:
            return diconew
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        # train.return_pandas().to_csv(f'CVAE_train_{self.argdict["dataset"]}.tsv', sep='\t')
        # fds
        return train

        # Creating new dataset
