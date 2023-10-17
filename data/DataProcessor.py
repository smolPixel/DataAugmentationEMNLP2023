from torch.utils.data import Dataset
import os, io
import numpy as np
import json
import pandas as pd
import torch
from collections import defaultdict, OrderedDict, Counter
from nltk.tokenize import TweetTokenizer, sent_tokenize
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import BertTokenizer
import math
import ast
import pickle
# import bcolz
import torch
import copy
import random


class OrderedCounter(Counter, OrderedDict):
	"""Counter that remembers the order elements are first encountered"""
	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)
def sample_class(df, i, prop, argdict):
	"""Sample the class i from the dataframe, with oversampling if needed. """
	size_class=len(df[df['label'] == i])
	ds=argdict['dataset_size'] if argdict['dataset_size']!=0 else len(df)
	num_sample_tot=math.ceil(ds * prop)
	#Sample a first time
	num_sample = min(num_sample_tot, size_class)
	sample = df[df['label'] == i].sample(n=num_sample)
	num_sample_tot-=num_sample
	while num_sample_tot!=0:
		num_sample=min(num_sample_tot, size_class)
		sampleTemp=df[df['label'] == i].sample(n=num_sample)
		sample = pd.concat([sample, sampleTemp])
		num_sample_tot-=num_sample
	return sample



def get_dataFrame(argdict):
	"""Get the dataframe for the particular split. If it does not exist: create it"""
	task=argdict['dataset']
	create_train=False
	dfVal = pd.read_csv(f'data/{task}/dev.tsv', sep='\t')
	dfTest=pd.read_csv(f'data/{task}/test.tsv', sep='\t')
	dfTrain=pd.read_csv(f'data/{task}/train.tsv', sep='\t').dropna(axis=1)
	# pd.set_option('display.max_rows', None)
	# pd.set_option('display.max_columns', None)
	# pd.set_option('display.width', None)
	# pd.set_option('display.max_colwidth', -1)
	#Sampling balanced data
	#We always oversample data to eliminate the unbalance factor from the DA algos, as the assumption is that DA is going to be more efficient if the data is unbalanced
	print(list(dfTrain))
	if argdict['dataset_size']==0:
		max_size = dfTrain['label'].value_counts().max()
		prop=max_size/len(dfTrain)
	else:
		prop=1/len(argdict['categories'])
	NewdfTrain=sample_class(dfTrain, 0, prop, argdict)
	for i in range(1, len(argdict['categories'])):
		prop = len(dfTrain[dfTrain['label'] == i]) / len(dfTrain)
		# TODO HERE
		prop = 1 / len(argdict['categories'])
		NewdfTrain=pd.concat([NewdfTrain ,sample_class(dfTrain, i, prop, argdict)])
	dfTrain=NewdfTrain
	# Path(pathTrain).mkdir(parents=True, exist_ok=True)
	# dfTrain.to_csv(f"{pathTrain}/train.tsv", sep='\t')
	# fdasklj
	print(f"Length of the dataframe {len(dfTrain)}")
	if argdict['fix_dataset']:
		dfTrain.to_csv(f'Selecteddata/{task}/{argdict["dataset_size"]}/train.tsv', sep='\t')

	return dfTrain, dfVal, dfTest





def initialize_dataset(argdict):
	if argdict['tokenizer'] == "tweetTokenizer":
		tokenizer = TweetTokenizer()
	elif argdict['tokenizer']=="PtweetTokenizer":
		tokenizer= TweetTokenizer()
	elif argdict['tokenizer'] == "mwordPiece":
		tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
		num_token_add = tokenizer.add_tokens(['<bos>', '<eos>', '<unk>', '<pad>'], special_tokens=True)  ##This line is updated
	else:
		raise ValueError("Incorrect tokenizer")


	if argdict['dataset'] in ['SST2']:
		from data.SST2.SST2Dataset import SST2_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			try:
				with open(f'/data/rali6/Tmp/piedboef/Models/DACon/{language}_vocab.pickle', 'rb') as f:
					vocab=pickle.load(f)
			except:
				vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
				vocab.set_default_index(vocab["<unk>"])
				with open(f'/data/rali6/Tmp/piedboef/Models/DACon/{language}_vocab.pickle', 'wb') as f:
					vocab=pickle.dump(vocab, f)
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=SST2_dataset(train, tokenizer, vocab, argdict)
		dev=SST2_dataset(dev, tokenizer, vocab, argdict)
		test=SST2_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test

	elif argdict['dataset'] in ['TREC6']:
		from data.TREC6.TREC6Dataset import TREC6_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			try:
				with open(f'/data/rali6/Tmp/piedboef/Models/DACon/{language}_vocab.pickle', 'rb') as f:
					vocab=pickle.load(f)
			except:
				vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
				vocab.set_default_index(vocab["<unk>"])
				with open(f'/data/rali6/Tmp/piedboef/Models/DACon/{language}_vocab.pickle', 'wb') as f:
					vocab=pickle.dump(vocab, f)
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=TREC6_dataset(train, tokenizer, vocab, argdict)
		dev=TREC6_dataset(dev, tokenizer, vocab, argdict)
		test=TREC6_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['Irony']:
		from data.Irony.IronyDataset import Irony_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
		vocab.set_default_index(vocab["<unk>"])
		train=Irony_dataset(train, tokenizer, vocab, argdict)
		dev=Irony_dataset(dev, tokenizer, vocab, argdict)
		test=Irony_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['IronyB']:
		from data.Irony.IronyDataset import Irony_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
		vocab.set_default_index(vocab["<unk>"])
		train=Irony_dataset(train, tokenizer, vocab, argdict)
		dev=Irony_dataset(dev, tokenizer, vocab, argdict)
		test=Irony_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['FakeNews']:
		from data.FakeNews.FakeNewsDataset import FakeNews_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
		vocab.set_default_index(vocab["<unk>"])
		train=FakeNews_dataset(train, tokenizer, vocab, argdict)
		dev=FakeNews_dataset(dev, tokenizer, vocab, argdict)
		test=FakeNews_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['MPhasisDe']:
		from data.MPhasisDe.MPhasisDeDataset import MPhasisDe_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
		vocab.set_default_index(vocab["<unk>"])
		train=MPhasisDe_dataset(train, tokenizer, vocab, argdict)
		dev=MPhasisDe_dataset(dev, tokenizer, vocab, argdict)
		test=MPhasisDe_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['SB10k']:
		from data.SB10k.SB10kDataset import SB10k_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			with open(f'/data/rali6/Tmp/piedboef/Models/DACon/de_vocab.pickle', 'rb') as f:
				vocab=pickle.load(f)
			print("loaded pretrained vocab")
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=SB10k_dataset(train, tokenizer, vocab, argdict)
		dev=SB10k_dataset(dev, tokenizer, vocab, argdict)
		test=SB10k_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['MPhasisFr']:
		from data.MPhasisFr.MPhasisFrDataset import MPhasisFr_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
		vocab.set_default_index(vocab["<unk>"])
		train=MPhasisFr_dataset(train, tokenizer, vocab, argdict)
		dev=MPhasisFr_dataset(dev, tokenizer, vocab, argdict)
		test=MPhasisFr_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['CLS']:
		from data.CLS.CLSDataset import CLS_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			with open(f'/data/rali6/Tmp/piedboef/Models/DACon/fr_vocab.pickle', 'rb') as f:
				vocab=pickle.load(f)
			print("loaded pretrained vocab")
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=CLS_dataset(train, tokenizer, vocab, argdict)
		dev=CLS_dataset(dev, tokenizer, vocab, argdict)
		test=CLS_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['ko3i4k']:
		from data.ko3i4k.ko3i4kDataset import ko3i4k_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			with open(f'/data/rali6/Tmp/piedboef/Models/DACon/ko_vocab.pickle', 'rb') as f:
				vocab=pickle.load(f)
			print("loaded pretrained vocab")
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=ko3i4k_dataset(train, tokenizer, vocab, argdict)
		dev=ko3i4k_dataset(dev, tokenizer, vocab, argdict)
		test=ko3i4k_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['koHateSpeech']:
		from data.koHateSpeech.koHateSpeech import koHateSpeech_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			with open(f'/data/rali6/Tmp/piedboef/Models/DACon/ko_vocab.pickle', 'rb') as f:
				vocab=pickle.load(f)
			print("loaded pretrained vocab")
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=koHateSpeech_dataset(train, tokenizer, vocab, argdict)
		dev=koHateSpeech_dataset(dev, tokenizer, vocab, argdict)
		test=koHateSpeech_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['Swahili']:
		from data.Swahili.SwahiliDataset import Swahili_dataset
		#Textual dataset


		train, dev, test=get_dataFrame(argdict)
		if argdict['tokenizer']=='PtweetTokenizer':
			with open(f'/data/rali6/Tmp/piedboef/Models/DACon/sw_vocab.pickle', 'rb') as f:
				vocab=pickle.load(f)
			print("loaded pretrained vocab")
		else:
			vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
			vocab.set_default_index(vocab["<unk>"])
		train=Swahili_dataset(train, tokenizer, vocab, argdict)
		dev=Swahili_dataset(dev, tokenizer, vocab, argdict)
		test=Swahili_dataset(test, tokenizer, vocab, argdict)
		argdict['input_size']=train.vocab_size

		return train, dev, test
	elif argdict['dataset'] in ['MNIST']:
		#Image dataset
		from data.MNIST.MNIST_dataset import MNIST_dataset
		train = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
		test = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(),
									  download=False)
		train, dev=torch.utils.data.random_split(train, [55000, 5000])
		train=MNIST_dataset(train)
		dev=MNIST_dataset(dev)
		test=MNIST_dataset(test)
		argdict['input_size']=784
		argdict['need_embedding'] = False
		return train, dev, test
	else:
		raise ValueError("dataset not found")

def createFolders(argdict):
	"""Create all folders necessary to the runs so you dont have to create them yourself"""
	"""The specific folders that need to be created are in GeneratedData, SelectedData and ../data"""

	num=checkFoldersGenerated(argdict)
	argdict['numFolderGenerated']=num
	Path(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{num}").mkdir(parents=True, exist_ok=True)
	Path(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}").mkdir(parents=True, exist_ok=True)
	Path(f"{argdict['pathDataAdd']}/data/{argdict['classifier']}/{argdict['dataset']}").mkdir(parents=True, exist_ok=True)
	Path(f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}").mkdir(parents=True, exist_ok=True)
	return argdict

def initialize_dataset_from_dataframe(argdict, dataframe):
	train=dataframe
	allsentences = list(train['sentence'])

	tokenizer = TweetTokenizer()
	# for sentence in allsentences:
	#     try:
	#         tokenizer.tokenize(sentence)
	#     except:
	#         print(sentence)
	#         if sentence!=sentence:
	#             print("babibi")
	#         fdsa
	allsentences = [tokenizer.tokenize(sentence) for sentence in allsentences if sentence == sentence]
	# print(allsentences)
	vocab = build_vocab_from_iterator(allsentences, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
	vocab.set_default_index(vocab["<unk>"])
	train = ds_DAControlled(train, tokenizer, vocab, argdict)
	# dev = ds_DAControlled(dev, tokenizer, vocab, argdict, dev=True)
	return train


def separate_per_class(dataset):
	"""Separate a dataset per class"""
	num_class=len(dataset.argdict['categories'])
	datasets=[copy.deepcopy(dataset) for _ in range(num_class)]
	# print(datasets)
	for ind, ex in dataset.data.items():
		lab=ex['label']
		for i, ds in enumerate(datasets):
			if i!=lab:
				ds.data.pop(ind)
	for ds in datasets:
		ds.reset_index()
	return datasets



class ds_DAControlled(Dataset):


	def __init__(self, data, tokenizer=None, vocab=None, argdict=None, dev=False, dataset_parent=None, from_dict=False):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.argdict=argdict
		if from_dict:
			self.data=data
			self.max_len=dataset_parent.max_len
			self.vocab=dataset_parent.vocab
			self.tokenizer=dataset_parent.tokenizer
		else:
			self.data = {}
			self.max_len = argdict['max_seq_length']
			if self.max_len==0:
				find_max_len=True
			else:
				find_max_len=False
			self.vocab = vocab
			self.argdict['vocab_size']=len(self.vocab)
			self.tokenizer=tokenizer
			# self.pad_idx = self.vocab['<pad>']
			self.max_len_label=0
			self.max_len_words=0
			self.num_sentences=0
			self.len_sentence=0
			index=0
			for i, row in data.iterrows():
				#For sentence VAE
				# sentences_sep=row['sentences'].strip().split('.')
				# print(sentences_sep)
				# sentences_sep=[sent.replace('.',' . ') for sent in sent_tokenize(row['sentence'].strip().replace(' . ', '.'))]
				#TODO CHECK IF THIS HERE DOES SOMETHING IMPORTANT
				sentences_sep=[vocab(self.tokenizer.tokenize("<bos> "+sent+" <eos>")) for sent in row['sentence']]
				if len(sentences_sep)>self.num_sentences:
					self.num_sentences=len(sentences_sep)
				# if row['sentence'] in ['.', '']:
				#     continue
				if self.len_sentence<max([len(sent) for sent in sentences_sep]):
					self.len_sentence=max([len(sent) for sent in sentences_sep])
				tokenized_text = self.tokenizer.tokenize("<bos> " + row['sentence'] + " <eos>")
				if find_max_len and self.max_len<len(tokenized_text):
					self.max_len=len(tokenized_text)
				input = np.array(vocab(tokenized_text))
				# tokenized_text=self.tokenizer.tokenize("<bos> "+row['sentence']+" <eos>")
				# sentence_max_len=" ".join(row['sentences'].split(' ')[:self.max_len])
				# output=np.array(vocab(tokenized_labels))
				# if len(output)>self.max_len_label:
				#     self.max_len_label=len(output)
				self.data[index] = {'input': input, 'label':row['label'], 'sentence':row['sentence'], 'augmented':False}
				index+=1

	@property
	def vocab_size(self):
		return len(self.vocab)

	@property
	def eos_idx(self):
		return self.vocab['<eos>']

	@property
	def pad_idx(self):
		return self.vocab['<pad>']

	@property
	def bos_idx(self):
		return self.vocab['<bos>']

	@property
	def unk_idx(self):
		return self.vocab['<unk>']

	def get_i2w(self):
		return self.vocab.get_itos()

	def __setitem__(self, key, value):
		self.data[key]=value

	def reset_index(self):
		new_dat={}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data=new_dat

	def get_random_example_from_class(self, classe):
		lab=None
		while lab!=classe:
			random_ex=random.randint(0, len(self.data)-1)
			dat=self.data[random_ex]
			lab=dat['label']
		return dat


	def tokenize(self, sentence):
		"Tokenize a sentence"
		return self.vocab(self.tokenizer.tokenize("<bos> "+sentence+" <eos>"))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		input = self.data[item]['input'][:self.max_len]
		label = self.data[item]['label']
		input=np.append(input, [self.pad_idx] * (self.max_len - len(input))).astype(int)
		# print("bitch")
		# sent_sep=np.array([np.append(sent, [self.pad_idx]*(self.len_sentence-len(sent))) for sent in self.data[item]['sentence_sep']], dtype=int)
		# print(sent_sep)
		# sent_sep=[np.append(sent, np.array(([self.pad_idx]*(self.len_sentence-len(sent)))) for sent in self.data[item]['sentence_sep'])]
		# print(sent_sep)
		# print(self.num_sentences)
		return {
			'index': item,
			'input': input,
			'label': label,
			'sentence':self.data[item]['sentence'],
			'augmented':self.data[item]['augmented'],
	}

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def arr_to_sentences(self, array):
		sentences=[]
		for arr in array:
			sent=self.vocab.lookup_tokens(arr.tolist())
			ss=""
			for token in sent:
				if token== "<bos>":
					continue
				if token =="<eos>":
					break
				ss+=f" {token}"
			sentences.append(ss)
		return sentences

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')

	def empty_exos(self):
		"""Remove all exemples from data"""
		self.data = {}