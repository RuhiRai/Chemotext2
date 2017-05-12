#/projects/sequence_analysis/vol3/tools/anaconda/bin
# -*- coding: utf-8 -*-
"""

Created on Tue Jul 05 16:49:41 2016

Code to generate data for training the EIM entity-entity relationship model.  The data
consists of a matrix where each row represents a ei-ej pair and the columns hold the
sikj value (along with ei and ej indices, known relationship date, case,...)

@author: cpschmitt
"""

import pandas as pd
from sets import Set
import numpy as np
import random
import gensim.models
import datetime
import string
import itertools
import math
import time
from datetime import date
import gc
import sys
import traceback
import os


class EntityPairInfo:
    # class to hold information about a pair of entities
    i1=-1
    i2=-1
    i1name=""
    i2name=""
    case=2
    reldate=None
    relposixdate=None
    datediff=None

    def __init__(self,i1,i2,i1name,i2name,case=2):
        self.i1=i1
        self.i2=i2
        self.i1name=i1name
        self.i2name=i2name
        self.case=case

def status(msg,fstatus):
    fstatus.write(msg)
#    fstatus.flush()
#    os.fsync(fstatus.fileno())
#    print msg


def getmapkeybyname(el, em):
    # given name of two entities, el and em, return a key to use for a dictionary.
    # names should be in lower case
    if el<em:
        return el+":"+em
    else:
        return em+":"+el   
    
def getpriormonth(curdate, nummonths):
    # return a number of month before a date object, assumes day is 1.
    retmonth=curdate.month-nummonths
    retyear=curdate.year
    while retmonth<=0:
        retmonth=retmonth+12
        retyear=retyear-1
    retdate=datetime.date(retyear,retmonth,1)
    return retdate

def getpriormonth_range(curdate, plow, phigh):
    # return a random number of months before a date object, assumes day is 1
    if plow==phigh:
        return getpriormonth(curdate, plow)
    else:
        rnum=(int)(plow+(phigh-plow)*random.random())
        return getpriormonth(curdate, rnum)
    
def genmodelname(mddate, mdduration):
    # get the model for a given date and duration and the evaluation date. Date are datetime.dates, duration>=1
    if mdduration==1:
        wemodelname=fwemodels+"month/pmc-"+str(mddate.year)+"-"+str(mddate.month)+".w2v"
    elif mdduration==2:
        wemodelname=fwemodels+"2month/pmc-"+str(mddate.year)+"-"+str(mddate.month)+".w2v"
    elif mdduration==12:
        wemodelname=fwemodels+"year/pmc-"+str(mddate.year)+"-"+str(mddate.month)+".w2v"
    elif mdduration==24:
        wemodelname=fwemodels+"2year/pmc-"+str(mddate.year)+"-"+str(mddate.month)+".w2v"
    elif mdduration==36:
        wemodelname=fwemodels+"3year/pmc-"+str(mddate.year)+"-"+str(mddate.month)+".w2v"
    return wemodelname

def getsimilarity(entity1, entity2, wemodel):
    # get the similarity between two entities indices, assumings all names are lower case
    if entity1 in wemodel.vocab and entity2 in wemodel.vocab:
        return wemodel.similarity(entity1,entity2)
    return 0.0    

def getEntitiesInModel(entitylist, modeldate, modelduration):
    # return a new list holding only entities in the semantic similiarity model for the given date and duration
    modelname=genmodelname(getpriormonth(modeldate,1), modelduration)
    model = gensim.models.Word2Vec.load(modelname)
    newlist=[]
    for i in range(0,len(entitylist)):
        if entitylist[i] in model.vocab:
            newlist.append(entitylist[i])
    return newlist


##################################
##################################
if len(sys.argv)<7:
    print 'Arguments evalyear evalmonth premonthslow premonthshigh duration'
evalyear=(int)(sys.argv[1])            # e,g, 2011
evalmonth=(int)(sys.argv[2])           # 1-based, e.g. 3 for march
premonthslow=(int)(sys.argv[3])        # >=1
premonthshigh=(int)(sys.argv[4])       # >=premonthslow
modelduration=(int)(sys.argv[5])      # >=1
fpredrelations=(sys.argv[6])

# Input data
fknownrelations = "/projects/mipseq/chemotext2/IntActKinaseP53Pairs.csv"
#fpredrelations = "/projects/mipseq/chemotext2/modelV8_2009/modelv8"
fwemodels = "/projects/stars/var/chemotext/w2v/gensim/cumulative/"
fsynonyms="/projects/mipseq/chemotext2/HGNCGeneSynonyms.csv"

# get evaluation date (date of word embedding generation)
evaldate = datetime.date(evalyear,evalmonth,1)      # year, month, day
evaldateposix = time.mktime(evaldate.timetuple())   # date to unix posix time

# create dataset file name
fpredrelations=fpredrelations+"_e"+str(evalyear)+str(evalmonth)
fpredrelations=fpredrelations+"_p"+str(premonthslow)+"-"+str(premonthshigh)
fpredrelations=fpredrelations+"_d"+str(modelduration)
fstatusrelations=fpredrelations+"_status"
fstatus=open(fstatusrelations+".txt","w",buffering=1)
status('Starting; evaluation date '+str(evaldate)+', poxis '+str(evaldateposix)+"\n",fstatus)

# read in entity interaction data
status("Reading interaction data\n",fstatus)
dfpairlist=pd.read_csv(fknownrelations)

# get list of entities and convert them all to a sorted lower case list 
fstatus.write("Setting up lists\n")
entityset=Set(dfpairlist['GeneA'].tolist())
entityset=entityset.union(dfpairlist['GeneB'].tolist())
entitylist=sorted(entityset)
entitylist=[x.lower() for x in (entitylist)]

# update the list of entities to not include entities that aren't in the most recent
#   semantic model
numorigentities=len(entitylist)
entitylist=getEntitiesInModel(entitylist,evaldate,modelduration)
numentities=len(entitylist)
numknownrelations=len(dfpairlist)
samplerate = 2.0*numknownrelations/(numentities*numentities)
status("Num original entities "+str(numorigentities)+"\n",fstatus)
status("Num entities "+str(numentities)+"\n",fstatus)
status("Num known relations "+str(numknownrelations)+"\n",fstatus)
status("Sampling rate "+str(samplerate)+"\n",fstatus)


# generate a map from each pair of entity names to information on their relationship
status("Generating info on entity pairs\n",fstatus)
pinfomap={}
for ei in range(numentities):
    for ej in range(ei+1,numentities):
        pinfomap[getmapkeybyname(entitylist[ei],entitylist[ej])]=EntityPairInfo(ei,ej,entitylist[ei],entitylist[ej])

# for each pair of entities, get the earliest known relationship
status("Generating earliest dates on entity pairs\n",fstatus)
for row in dfpairlist.iterrows():    
    entitya=row[1]['GeneA'].lower()
    entityb=row[1]['GeneB'].lower()
    if (entitya not in entitylist or entityb not in entitylist):
        continue
    pinfo=pinfomap[getmapkeybyname(row[1]['GeneA'].lower(),row[1]['GeneB'].lower())]
    if pinfo.relposixdate is None or pinfo.relposixdate>row[1]['UTCDate']:
        pinfo.relposixdate = row[1]['UTCDate']                      
        pinfo.reldate = date.fromtimestamp(pinfo.relposixdate)
        pinfo.datediff = evaldateposix - pinfo.relposixdate
        if pinfo.datediff>0:
            pinfo.case=0
        else:
            pinfo.case=1

# now build a map from model name to which entity pairs will use that model
modelmap={}
for pinfo in pinfomap.itervalues():
#    if pinfo.reldate is not None:
    if pinfo.case == 0:
        modeldate=getpriormonth_range(pinfo.reldate, premonthslow, premonthshigh)
    else:
        modeldate=getpriormonth(evaldate,1)
    modelname=genmodelname(modeldate, modelduration)
    if modelname not in modelmap:
        modelmap[modelname]=[]
    modelmap[modelname].append(pinfo)

# write header column names to output
status("Generating column names\n",fstatus)
hdr="entitya,entityb,case,relposixdate,reldate,metimediff,"
hdr+=",".join(item for item in entitylist)
fpred=open(fpredrelations+".txt","w")
fpred.write(hdr+"\n")

# main loop for each model, output a row for each pair using the modelmap to avoid loading/reloading models
status("Generate sijk data\n",fstatus)
numskipped=0
numprocessed=0
numcases=[0]*3
for modelname in modelmap:
    model = gensim.models.Word2Vec.load(modelname)
    for eijinfo in modelmap[modelname]:
        sikj=[0]*numentities

        numprocessed+=1
        if (numprocessed % 50 == 0):
            status("On "+str(numprocessed)+"\n",fstatus)

        # sample case=2 as these are over represented
        if eijinfo.case==2:
            if random.random()>samplerate:
                numskipped=numskipped+1
                continue

        # build up ei-ek list of semantic similarity
        for ek in range(0,numentities):
            if ek==eijinfo.i1:               # if ei==ek or ej==ek
                sikj[ek]=float('NaN')
            else: 
                eikinfo=pinfomap[getmapkeybyname(eijinfo.i1name,entitylist[ek])]
                sikj[ek]=getsimilarity(eikinfo.i1name,eikinfo.i2name,model)
                if eikinfo.case==1 or eikinfo.case==2:                       # no relation date or relation date after evaluate
                    sikj[ek]=-sikj[ek]
    
        # now output sij information
        strbuff = eijinfo.i1name+","+eijinfo.i2name+","+str(eijinfo.case)+","+str(eijinfo.relposixdate)+","+str(eijinfo.reldate)+","+str(eijinfo.datediff)+","
        strbuff += ','.join([str(x) for x in sikj])
        strbuff +="\n"
        fpred.write(strbuff)
        numcases[eijinfo.case]=numcases[eijinfo.case]+1

fpred.close()
status("Number eijs skipped "+str(numskipped)+"\n",fstatus)
status("Num cases "+str(numcases)+"\n",fstatus)
status("Done\n",fstatus)
fstatus.close()





