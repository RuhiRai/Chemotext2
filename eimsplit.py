"""
Created on Thu Nov 03 16:29:29 2016

Module for building a model for predicting entity entity interactions from eimdatagenerator

Each row of input data set contains an ei, ej pair and the values for all eijk
In addition each row holds the indicated relationship date between ei and ej
and a Case indicator where

case:
1=relationship date < evaluation date
2=relationship date >= evaluation date
3=no relationship date





@author: cpschmitt
"""
import pandas as pd
from sets import Set
import numpy as np
import random
import gensim.models
import time
import datetime
import string
import itertools
import math
import sys


trainper=0.8
valper=-100   # set < 0 to turn off

fmodeldata = (sys.argv[1])

ftraindata = fmodeldata+".train.txt"
ftrain=open(ftraindata,"w")

ftestdata = fmodeldata+".test.txt"
ftest=open(ftestdata,"w")

if valper > 0:
    fvalidatedata = fmodeldata+".validate.txt"
    fvalidate=open(fvalidatedata,"w")


with open(fmodeldata) as frdr:
    hdr=next(frdr)
    ftrain.write(hdr)
    ftest.write(hdr)
    if valper>0:
        fvalidate.write(hdr)
    for line in frdr:
        r=random.random()
        if r<trainper:
            ftrain.write(line)
        elif valper>0 and r<valper:
            fvalidate.write(line)
        else:
            ftest.write(line)

ftrain.close();
if valper>0:
    fvalidate.close();
ftest.close();
