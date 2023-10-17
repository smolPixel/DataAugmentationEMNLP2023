import pandas as pd
import json
import re

dataset="SST2"
dataset_size_wanted=500
split=1
num_var=15

file=json.load(open(f'{dataset}/{dataset_size_wanted}/Output.json', 'r'))
# print(file['answer'])
# print(len(file['answer']))

if split>1:
	#In this case each answer is for one entry
	#We have 10 data points per entry
	random_seed=0
	df = pd.DataFrame(columns=['sentence', 'label'])
	og = pd.read_csv(f'../../Selecteddata/{dataset}/{dataset_size_wanted}/train.tsv', sep='\t')
	dataset_size=len(og)
	for counter, dat in enumerate(file['answer']):
		if counter%dataset_size==0 and counter>0:
			#We have reached the end of a random seed
			# print(len(df))
			# assert len(df)==dataset_size*split
			df.to_csv(f'{dataset}/{dataset_size_wanted}/{random_seed}.tsv', sep='\t')
			random_seed+=1
			df = pd.DataFrame(columns=['sentence', 'label'])
		dat=dat.replace('<ol><li>', '').replace('</li></ol>', '').replace('<p>', '').replace('</p>', '')
		dat=dat.split('</li><li>')
		if len(dat)!=split:
			continue
		for da in dat:
			ll=len(df)
			df.at[ll, 'sentence']=da
			df.at[ll, 'label']=og.at[counter%dataset_size, 'label']

