"""Class for augmentation strategies"""
import pandas as pd

class augmentator():

	def __init__(self, argdict):
		self.argdict=argdict
		# self.dataset=dataset
		# self.train=pd.read_csv(f"{argdict['path']}/data/SST-2/train.tsv", sep='\t', index_col=0)
		# self.dev=pd.read_csv(f"{argdict['path']}/data/SST-2/dev.tsv", sep='\t', index_col=0)
		if self.argdict['algo']=='dummy':
			from AugmentStrat.NoAug import NoAug
			self.algo=NoAug(self.argdict)
		elif self.argdict['algo']=='Copie':
			from AugmentStrat.Copie import Copie
			self.algo=Copie(self.argdict)
		elif self.argdict['algo']=='EDA':
			from AugmentStrat.EDA import EDA
			self.algo=EDA(self.argdict)
		elif self.argdict['algo'].lower()=='backtranslation':
			from AugmentStrat.BackTranslation import BackTranslation
			self.algo=BackTranslation(self.argdict)
		elif self.argdict['algo'].lower()=='backtranslationt5':
			from AugmentStrat.BackTranslationT5 import BackTranslationT5
			self.algo=BackTranslationT5(self.argdict)
		elif self.argdict['algo']=='CATGAN':
			from AugmentStrat.CATGAN import CatGan
			self.algo=CatGan(self.argdict)
		elif self.argdict['algo']=='VAE':
			from AugmentStrat.VAE import VAE
			self.algo=VAE(self.argdict)
		elif self.argdict['algo']=='VAE_Link':
			from AugmentStrat.VAE_Link import VAE_Link
			self.algo=VAE_Link(self.argdict)
		elif self.argdict['algo']=='PVAE':
			from AugmentStrat.PVAE import PVAE
			self.algo=PVAE(self.argdict)
		elif self.argdict['algo']=='PAE':
			from AugmentStrat.PAE import PAE
			self.algo=PAE(self.argdict)
		elif self.argdict['algo']=='VAE_Par':
			from AugmentStrat.VAE_Par import VAE_Par
			self.algo=VAE_Par(self.argdict)
		elif self.argdict['algo'].lower()=='transformers_vae':
			from AugmentStrat.transformerIntoVAE import VAE
			self.algo=VAE(self.argdict)
		elif self.argdict['algo']=='CVAE':
			from AugmentStrat.CVAE import CVAE
			self.algo=CVAE(self.argdict)
		elif self.argdict['algo']=='CBERT':
			from AugmentStrat.CBERT import CBERT
			self.algo=CBERT(self.argdict)
		elif self.argdict['algo']=='mCBERT':
			from AugmentStrat.mCBERT import mCBERT
			self.algo=mCBERT(self.argdict)
		elif self.argdict['algo']=='T5Par':
			from AugmentStrat.T5 import T5Paraphrase
			self.algo=T5Paraphrase(self.argdict)
		elif self.argdict['algo']=='T5ParQuora':
			from AugmentStrat.T5Quora import T5ParaphraseQuora
			self.algo=T5ParaphraseQuora(self.argdict)
		elif self.argdict['algo']=='T5BT':
			from AugmentStrat.T5BT import T5ParaphraseBT
			self.algo=T5ParaphraseBT(self.argdict)
		#DummyChatGpt is simply a first pass to retrieve the data in form useable for the automatisation script
		elif self.argdict['algo']=='ChatGPTDummy':
			from AugmentStrat.ChatGPTDummy import ChatGPTDummy
			self.algo=ChatGPTDummy(self.argdict)
		elif self.argdict['algo']=='ChatGPT':
			from AugmentStrat.ChatGPT import ChatGPT
			self.algo=ChatGPT(self.argdict)
		elif self.argdict['algo']=='ChatGPTDescriptionDummy':
			from AugmentStrat.ChatGPTDescriptionDummy import ChatGPTDescriptionDummy
			self.algo=ChatGPTDescriptionDummy(self.argdict)
		elif self.argdict['algo']=='ChatGPTDescription':
			from AugmentStrat.ChatGPTDescription import ChatGPTDescription
			self.algo=ChatGPTDescription(self.argdict)
		elif self.argdict['algo']=='ChatGPTKShotDummy':
			from AugmentStrat.ChatGPTKShotDummy import ChatGPTKShotDummy
			self.algo=ChatGPTKShotDummy(self.argdict)
		elif self.argdict['algo']=='ChatGPTKshot':
			from AugmentStrat.ChatGPTKShot import ChatGPTKshot
			self.algo=ChatGPTKshot(self.argdict)
		elif self.argdict['algo']=='GPT':
			from AugmentStrat.GPT import GPT
			self.algo=GPT(self.argdict)
		elif self.argdict['algo']=='BART':
			from AugmentStrat.BART import BART
			self.algo=BART(self.argdict)
		elif self.argdict['algo']=='AEDA':
			from AugmentStrat.AEDA import AEDA
			self.algo=AEDA(self.argdict)
		elif self.argdict['algo']=='Perfect':
			from AugmentStrat.Perfect import Perfect
			self.algo=Perfect(self.argdict)
		elif self.argdict['algo']=='PerfectEDA':
			from AugmentStrat.PerfectEDA import PerfectEDA
			self.algo=PerfectEDA(self.argdict)
		else:
			raise ValueError("Augmentator not found")

	def augment(self, train, dev=None):
		"""Augment takes the dataframes data/train.tsv et dev.tsv and drop augmented train and dev in /Tmp/"""
		if self.argdict['split']==0:
			return train
		else:
			return self.algo.augment(train, dev)
		# self.dev.to_csv(f"/Tmp/dev_{self.argdict['dataset']}.tsv", sep='\t')
	def normalize_punctuation(self, train, dev):
		from AugmentStrat.EDA_strat.EDA import get_only_chars

		dftrain=train.return_pandas()
		for i, ex in dftrain.iterrows():
			# print(ex)
			sent=ex['sentence']
			train.data[i]['sentence']=get_only_chars(sent)

		dfDev = dev.return_pandas()
		for i, ex in dfDev.iterrows():
			# print(ex)
			sent = ex['sentence']
			dev.data[i]['sentence'] = get_only_chars(sent)
			# fds
		# fds
		# print(dev.data)
		# fds
		return train, dev

	def augment_false(self, train, n):
		"""Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
		return self.algo.augment_false(train, n)

	def augment_false_dataset(self, train, n):
		print(train)
		trainAll=pd.read_csv(f"data/SST-2/train.tsv", sep='\t')
		print(trainAll)

	def augment_doublons(self, train, n):
		"""Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
		return self.algo.augment_doublons(train, n)

	def augment_doublons_algo(self, train, n):
		"""Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
		return self.algo.augment_doublons_algo(train, n)

	def transform_data(self, train, dev):
		"""Passes all the data through the DA algorithm"""
		return self.algo.transform_data(train, dev)
