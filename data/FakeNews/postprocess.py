import pandas as pd
import re


file='dev.tsv'

df=pd.read_csv(file, sep='\t')
for i, row in df.iterrows():
	ss=re.sub('- The New York Times', '', row['sentence'])
	ss=re.sub('- Breitbart', '', ss)
	ss=re.sub('\\| New Eastern Outlook', '', ss)
	df.at[i, 'sentence']=ss

df.to_csv(file, sep='\t')