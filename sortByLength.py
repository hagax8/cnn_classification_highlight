import numpy as np


inputFile = 'chembl_smiles_processed.rand.csv'
sample = []
with open(inputFile, 'r') as f:
    for line in f:
        sample.append(line.strip().split())

sample.sort(key=lambda x: len(x[1]), reverse=True)
np.savetxt('chembl_smiles_processed.rand.sorted.csv',sample,fmt='%s')
