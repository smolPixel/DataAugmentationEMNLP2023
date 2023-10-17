"""Simple strategy that consists in copying the dataset twice"""
"""EDA augmentation"""
import pandas as pd

class ChatGPTDescription():

    def __init__(self, argdict):
        self.argdict=argdict
        self.num_classes=len(argdict['categories'])

    def augment(self, train, return_dict):
        bs = 10
        #We put some safety measure in place since we're dealing with money
        # training_set = pd.read_csv(f"Temp/data_{i}.tsv", sep='\t', index_col=0)
        train_pandas=train.return_pandas()
        df=pd.read_csv(f'AugmentStrat/ChatGPTData/{self.argdict["dataset"]}/Description/Generated_senteces.tsv', sep='\t', header=0)
        for i in range(self.num_classes):
            num_to_sample=len(train_pandas[train_pandas['label']==i])*self.argdict['split']
            #Since we do oversampling:
            if num_to_sample>len(df[df['label']==i]):
                df_temp=df[df['label']==i].sample(n=num_to_sample, replace=True)
            else:
                df_temp = df[df['label'] == i].sample(n=num_to_sample)
            for i, tt in df_temp.iterrows():
                len_data = len(train)
                # print(tt['aug_sentence'])
                # print(tt['label'])
                train.data[len_data] = {'sentence': tt['aug_sentence'],
                                        'label': int(tt['label']),
                                        'input': train.tokenize(tt['aug_sentence']),
                                        'augmented': True}
        # else:
        #
        #     df=pd.read_csv(f'AugmentStrat/ChatGPTData/{self.argdict["dataset"]}/0/Generated_senteces.tsv', sep='\t')
        #     train_pandas=train.return_pandas()
        #     for i, tt in train_pandas.iterrows():
        #         ss=tt['sentence']
        #         # print(ss)
        #         ind=df[df['og_sentence']==ss.strip()]
        #         # print(ind)
        #         # print(ind['aug_sentence'].values[0])
        #         # a=ind['label'].values[0]
        #         # print(ind['label'])
        #         # print(a)
        #         # fds
        #         len_data = len(train)
        #         # print(item)
        #         print(ss)
        #         print(ind['aug_sentence'])
        #         train.data[len_data] = {'sentence':ind['aug_sentence'].values[0],
        #                              'label':ind['label'].values[0],
        #                              'input':train.tokenize(ind['aug_sentence'].values[0]),
        #                              'augmented':True}

        # rr=self.argdict['random_seed']
        # file_aug=pd.read_csv(f'AugmentStrat/ChatGPTData/{self.argdict["dataset"]}/{self.argdict["dataset_size"]}/{rr}.tsv', sep='\t', index_col=0)
        # print('------ CHECK THAT THESE DATA MAKE SENSE --------')
        # print(list(train.return_pandas()['sentence'])[-1])
        # print(list(file_aug['sentence'])[-self.argdict['split']])
        # for j, item in file_aug.iterrows():
        #     len_data = len(train)
        #     # print(item)
        #     train.data[len_data] = {'sentence':item['sentence'],
        #                          'label':item['label'],
        #                          'input':train.tokenize(item['sentence']),
        #                          'augmented':True}
        # train.return_pandas().to_csv(f"test_{self.argdict['random_seed']}.tsv", sep='\t')
        return train
