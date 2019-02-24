import subprocess
from sys import argv
import numpy as np
import random
import math
import os
import pandas as pd

inputFile = argv[1]
outputFile = argv[2]

df = pd.read_table(inputFile,header=None,sep=" ",names=['smiles','chembl_id'],usecols=[0,2])

df[['smiles']].to_csv(outputFile+".smiles",index=False,header=None)


cmdStandardize = ['/Applications/StructureRepresentationToolkit' +
                  '/bin/standardize',
                  '-v',
                  outputFile+".smiles",
                  '-o',
                  outputFile+'.stand.smiles',
                  '-c',
                  '"removeexplicitH..dearomatize..stripsalts..' +
                  'removefragment..clearstereo..wedgeclean..' +
                  'aromatize:basic..[O-][N+]=O>>O=N=O' +
                  '"']

process = subprocess.Popen(cmdStandardize, stdout=subprocess.PIPE)
print("done standardizing")

for line in process.stdout:
    print(line)

df[['stand_smiles']] = pd.read_table(outputFile+'.stand.smiles',
                                     header=None,sep=" ")
df.drop_duplicates(subset=['stand_smiles'], keep='first', inplace=True)
df[['chembl_id','stand_smiles']].to_csv(outputFile+'.stand.smiles.csv',sep=" ",
                                        header=None, index=False)
