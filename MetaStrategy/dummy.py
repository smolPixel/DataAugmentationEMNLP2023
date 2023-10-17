

class dummy_metastrat():

    def __init__(self, argdict, augmentator):
        """Does not change the augmentator"""
        self.augmentator=augmentator

    def augment(self, train, dev, test):
        return self.augmentator.augment(train, dev)

    def augment_false(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_false(train, n)


    def augment_doublons(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_doublons(train, n)

    def augment_doublons_algo(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_doublons_algo(train, n)

