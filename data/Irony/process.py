import pandas as pd


df=pd.read_csv('test.tsv', sep='\t')
df=df.rename(columns={'Tweet text':'sentence', 'Label':'label'})
df.to_csv('test.tsv', sep='\t', index=None)