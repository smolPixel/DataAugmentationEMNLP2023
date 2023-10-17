"""Simple strategy that consists in copying the dataset twice"""
"""EDA augmentation"""
import pandas

class Copie():

    def __init__(self, argdict):
        self.argdict=argdict
        self.num_classes=len(argdict['categories'])

    def augment(self, train, return_dict):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        if self.argdict['corrupt_data']>0:
            raise ValueError("NOT IMPLEMENTED CORRECTLY")
            nb_to_corrupt=num_points*split*self.argdict['corrupt_data']/len(self.argdict['categories'])
            print(f"corrputing {nb_to_corrupt} per categorie")
            nb_corrupt_per_class=[0]*self.num_classes

        diconew={}
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                # if self.argdict['corrupt_data'] > 0 and nb_corrupt_per_class[label] < nb_to_corrupt and label==0:
                #     label=1
                #     nb_corrupt_neg+=1
                # elif self.argdict['corrupt_data'] > 0 and nb_corrupt_pos < nb_to_corrupt and label==1:
                #     label=0
                #     nb_corrupt_pos+=1
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                sentAug=line
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug),
                                 'augmented':True}
                i+=1

        if return_dict:
            return diconew
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train

        # Creating new dataset
