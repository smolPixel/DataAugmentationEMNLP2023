"""EDA augmentation"""
import pandas
from AugmentStrat.EDA_strat.EDA import eda

class BartVAE():

    def __init__(self, argdict):
        self.argdict=argdict

    def augment(self, train, dev, return_dict=False):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)

        diconew={}
        #TODO ADD DATA IN FOLDER
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                # if self.argdict['corrupt_data'] > 0 and nb_corrupt_neg < nb_to_corrupt and label==0:
                #     label=1
                #     nb_corrupt_neg+=1
                # elif self.argdict['corrupt_data'] > 0 and nb_corrupt_pos < nb_to_corrupt and label==1:
                #     label=0
                #     nb_corrupt_pos+=1
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                sentAug=eda(line, self.argdict['alpha_sr'], self.argdict['alpha_ri'], self.argdict['alpha_rs'], self.argdict['alpha_rd'], 1)[0]
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug),
                                 'augmented':True}
                i+=1
        # if return_dict:
        #     return diconew
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train

        # Creating new dataset

    def normalize_punctuation(self, train, dev):
        from AugmentStrat.EDA_strat.EDA import get_only_chars
        print(train)
        fds

    def augment_false(self, train, n):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        diconew={}
        numFalsePos=0
        numFalseNeg=0

        #TODO ADD DATA IN FOLDER
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                if numFalsePos<n and label==1:
                    label=0
                    numFalsePos+=1
                elif numFalseNeg<n and label==0:
                    numFalseNeg+=1
                    label=1
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                sentAug=eda(line, 0.1, 0, 0, 0, 'en')
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug)}
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train

    def augment_doublons(self, train, n):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        # self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        diconew={}
        numFalsePos=0
        numFalseNeg=0
        print(n)

        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                if numFalsePos<n and label==1:
                    numFalsePos+=1
                    sentAug=line
                elif numFalseNeg<n and label==0:
                    numFalseNeg+=1
                    sentAug=line
                else:
                    sentAug = eda(line, 0.05, 0.05, 0.05, 0.05, 1)[0]

                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                # sentAug=eda(line, 0.1, 0.1, 0.1, 0.1, 'en')
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug)}
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.return_pandas().to_csv(f'test.tsv', sep='\t')
        return train


    def transform_data(self, train, dev):
        print("WARNING, THIS HAS ONLY BEEN PROGRAMMED FOR 1000 data - 1000 aug, AND WITH PERFECT AUG")
        split = self.argdict['split']  # Percent of data points added

        train_df=train.return_pandas()
        for i, ex in train_df.iterrows():
            line = ex['sentence']
            label = ex['label']
            # print('---')
            # print(len(line.split()))
            # print(line)
            # print("Bitch")
            sentAug = eda(line, 0.1, 0, 0, 0, 1)[0]
            train.data[i]['sentence']=sentAug
            if i==1000:
                break

        dev_df = dev.return_pandas()
        for i, ex in dev_df.iterrows():
            line = ex['sentence']
            label = ex['label']
            # print('---')
            # print(len(line.split()))
            # print(line)
            # print("Bitch")
            sentAug = eda(line, 0.1, 0, 0, 0, 1)[0]
            dev.data[i]['sentence'] = sentAug
        return train, dev