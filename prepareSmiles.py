import subprocess
from sys import argv
import numpy as np
import random
import math
import os

inputFile = 'chembl_smiles_processed.rand.csv'
size = int(argv[1])
jpgdir = argv[2]
SEED = 123
outputFile = jpgdir+"_temporary/out"
suffixList = ['train', 'val', 'test']

os.mkdir(jpgdir+"_temporary")
os.mkdir(jpgdir+"_train")
os.mkdir(jpgdir+"_val")
os.mkdir(jpgdir+"_test")

def wc(filename):
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])

linecount = wc(inputFile)

def random_sample(inputFile, size, linecount):
    with open(inputFile, 'r') as f:
        sample = []
        samplesmi = []
        random.seed(a=SEED)
        random_lines = sorted(random.sample(range(linecount), size),
                              reverse = True)
        lineID = random_lines.pop()
        for n, line in enumerate(f):
            if n == lineID:
                sample.append(line.rstrip().split()[0])
                samplesmi.append(line.rstrip().split()[1])
                if len(random_lines) > 0:
                    lineID = random_lines.pop()
                else:
                    break
    return sample,samplesmi

trainStart = 0
trainEnd = math.floor(size*0.50)
valStart = trainEnd
valEnd = valStart + math.floor(size*0.25)
testStart = valEnd
testEnd = size

sample,samplesmi = random_sample(inputFile, size, linecount)

sampledic = {}
samplesmidic = {}

sampledic['train'] = sample[trainStart:trainEnd]
sampledic['val'] = sample[valStart:valEnd]
sampledic['test'] = sample[testStart:testEnd]
samplesmidic['train'] = samplesmi[trainStart:trainEnd]
samplesmidic['val'] = samplesmi[valStart:valEnd]
samplesmidic['test'] = samplesmi[testStart:testEnd]


for suffix in suffixList:
    np.savetxt(outputFile+"_rand_"+str(size)+'_'+suffix,
               sampledic[suffix],fmt='%s')
    np.savetxt(outputFile+"_rand_"+str(size)+'_'+suffix+".smiles",
                samplesmidic[suffix],fmt='%s')


for suffix in suffixList:
    cmdStandardize = ['/Applications/StructureRepresentationToolkit/bin/standardize',
                      '-v',
                      outputFile+"_rand_"+str(size)+"_"+suffix+".smiles",
                      '-o',
                      outputFile+"_"+suffix,
                      '-c',
                      '"dearomatize..wedgeclean..aromatize:basic..removeexplicitH..stripsalts..removefragment..clearstereo..[O-][N+]=O>>O=N=O"']
    process = subprocess.Popen(cmdStandardize, stdout=subprocess.PIPE)
    print("done standardizing")
    for line in process.stdout:
        print(line)
    with open(outputFile+"_"+suffix, 'r') as f:
        for n, line in enumerate(f):
            print(line)
            cmdImage = ['/Applications/marvinbeans/bin/molconvert',
                        'jpeg:w170,h170,q98,#ffffff',
                        '--smiles',
                        str(line),
                        '-m',
                        '-o',
                        jpgdir+"_"+suffix+'/'+str(sampledic[suffix][n])+'.jpg']
            process = subprocess.Popen(cmdImage, stdout=subprocess.PIPE)
            for line in process.stdout:
                print(line)
