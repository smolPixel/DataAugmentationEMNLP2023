import pandas as pd

df=pd.read_csv('train.tsv', sep='\t')
print(len(df))

test=df.sample(frac=0.1, axis=0)
train=df.drop(index=test.index)
train=train.reset_index()
test=test.reset_index()

print(len(train))
print(len(test))

train.to_csv('train.tsv', sep='\t')
test.to_csv('test.tsv', sep='\t')