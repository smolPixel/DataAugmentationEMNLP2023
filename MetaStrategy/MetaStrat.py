"""Class for augmentation strategies"""
import pandas as pd

class meta_strat():

    def __init__(self, argdict, augmentator):
        self.argdict=argdict
        if argdict['meta_strategy']=="dummy":
            from MetaStrategy.dummy import dummy_metastrat
            self.metastrat=dummy_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=="doublons":
            from MetaStrategy.Doublons import doublons_metastrat
            self.metastrat=doublons_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='confidence':
            from MetaStrategy.Confidence import confidence_metastrat
            self.metastrat=confidence_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='influence':
            from MetaStrategy.InfluenceFunction import influenceFunction_metastrat
            self.metastrat=influenceFunction_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='disagreement':
            from MetaStrategy.Disagreement import disagreement_metastrat
            self.metastrat=disagreement_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='confidence_iter':
            from MetaStrategy.ConfidenceIter import confidence_iter_metastrat
            self.metastrat=confidence_iter_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='gradient':
            from MetaStrategy.Gradient import gradient_metastrat
            self.metastrat=gradient_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='classifier':
            from MetaStrategy.Classifier import classifier_metastrat
            self.metastrat=classifier_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='distance':
            from MetaStrategy.Distance import distance_metastrat
            self.metastrat=distance_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='cluster':
            from MetaStrategy.Cluster import cluster_metastrat
            self.metastrat=cluster_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='remove_og':
            from MetaStrategy.remove_og import remove_og_metastrat
            self.metastrat=remove_og_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='stack':
            from MetaStrategy.Stack import stack_metastrat
            self.metastrat=stack_metastrat(argdict, augmentator)
        elif argdict['meta_strategy']=='weak_lab':
            from MetaStrategy.WeakLabelling import weak_lab_metastrat
            self.metastrat=weak_lab_metastrat(argdict, augmentator)
        else:
            raise ValueError("Metastrat not found")

    def evaluate(self, example, train):
        return self.metastrat.evaluate(example, train)

    def prep_algo(self, train):
        return self.metastrat.prep_algo(train)

    def augment(self, train, dev, test):
        """Augment takes the dataframes data/train.tsv et dev.tsv and drop augmented train and dev in /Tmp/"""
        return self.metastrat.augment(train, dev, test)
        # self.dev.to_csv(f"/Tmp/dev_{self.argdict['dataset']}.tsv", sep='\t')

    def augment_false(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.metastrat.augment_false(train, n)

    def augment_false_dataset(self, train, n):
        num_pos_changed=0
        num_neg_changed=0
        train_pd=train.return_pandas()
        for i, row in train_pd.iterrows():
            if row['label']==0 and num_neg_changed<n:
                train.data[i]['label']=1
                num_neg_changed+=1
            if row['label']==1 and num_pos_changed<n:
                train.data[i]['label']=0
                num_neg_changed+=1
            if num_neg_changed==n and num_pos_changed==n:
                break
        return train

    def augment_doublons(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.metastrat.augment_doublons(train, n)

    def augment_doublons_dataset(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        num_pos_doublons = 0
        num_neg_doublons = 0
        train_pd = train.return_pandas()
        for i, row in train_pd.iterrows():
            if row['label'] == 0 and num_neg_doublons < n:
                train.data[i]['label'] = 1
                num_neg_changed += 1
            if row['label'] == 1 and num_pos_changed < n:
                train.data[i]['label'] = 0
                num_neg_changed += 1
            if num_neg_changed == n and num_pos_changed == n:
                break
        return train

    def augment_doublons_algo(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.metastrat.augment_doublons_algo(train, n)