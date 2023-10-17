import pandas as pd
import numpy as np

dataset='koHateSpeech'

df=pd.read_csv(f'{dataset}/train.tsv', sep='\t')

sents=list(df['sentence'])

print(f"num exo: {len(sents)}")
print(f"Num classes: {len(set(list(df['label'])))}")
print(f"Average len of sentence: {np.mean([len(ss) for ss in [ss.split(' ') for ss in sents]])}")