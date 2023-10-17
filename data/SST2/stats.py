import pandas as pd
import numpy as np
df=pd.read_csv('train.tsv', sep='\t')

sents=list(df['sentence'])
sents=[len(ss.split(' ')) for ss in sents]
print(np.mean(sents))
sents=[ll for ll in sents if ll<15]
print(np.mean(sents))