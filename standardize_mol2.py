import subprocess
from sys import argv
import numpy as np
import random
import math
import os
import pandas as pd

inputFile = argv[1]
outputFile = argv[2]


cmdStandardize = ['/Applications/StructureRepresentationToolkit' +
                  '/bin/standardize',
                  '-v',
                  inputFile,
                  '-o',
                  outputFile+'.stand.smiles',
                  '-c',
                  '"addexplicitH..removeexplicitH..dearomatize..stripsalts..' +
                  'removefragment..clearstereo..wedgeclean..' +
                  'aromatize:basic..[O-][N+]=O>>O=N=O' +
                  '"']


process = subprocess.Popen(cmdStandardize, stdout=subprocess.PIPE)
print("done standardizing")

for line in process.stdout:
    print(line)


stand_smiles = pd.read_table(outputFile+'.stand.smiles',
                                     header=None,sep=" ")
stand_smiles.drop_duplicates(keep='first', inplace=True)
stand_smiles.to_csv(outputFile+'.stand.smiles.csv',sep=" ",
                                        header=None, index=False)
