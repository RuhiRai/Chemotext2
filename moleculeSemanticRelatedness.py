# -*- coding: utf-8 -*-
"""

TODO: use per year word2vec models
TODO: adjust for P53 and synonyms in word embeddings calculations
TODO: read in a real word2vec model
TODO: improve the intact score calculation for known interactions (base it on methods and number of supporting evidence)

Created on Tue Jul 05 16:49:41 2016

Code to use semantic relatedness of small molecules to predict new 
molecule to molecule relationships.

Given a molecule $m$, this defines a semantic context $sc(m)$ as a $n_sc$ sized vector 
that maps the molecule to a semantic space.  This is built using an external program to 
construct the models. This program assumes the model is built and stored as a python 
Gensim word2vec model

Given a molecule $mi$, this outputs the value Ni and Ns and Ni/Ns.  Ns is the number
of other molecules $mj$ in the set $Smi$ which is the set of molecules
for which $s(mi,mj)$ is greater than a threshold. Ni is the number of molecules
in $Smi$ that have a known interaction with a given molecules (in our case, TP53).

input file:
dict1 = {'GeneA': genea, 'GeneB': geneb, 'PubMedId': pubmedid, 
         "Negative" : negflag, "IdMethodA": geneamethod, "IdMethodB": genebmethod,
         'UTCDate' : pubmeddate}


output file:
{'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Ni/Ns' : (ni/ns), 'Hit' : true/false}
where hit is true if no existing known relationship exists between the gene
and the target gene for a given evaluation date


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

class testmodel:
    simmap={}   # map from hash of genes to similarity score
    vocab=[]
    def __init__(self, infile):
        dfmodel=pd.read_csv(infile)
        self.vocab=dfmodel.GeneA.unique()
        for row in dfmodel.iterrows():
            genei = row[1]['GeneA']
            genej = row[1]['GeneB']  
            score = row[1]['Score']  
            self.simmap[self.gethash(genei,genej)]=score
    def similarity(self, genei, genej):
        hkey=self.gethash(genei,genej)
        if self.simmap.has_key(hkey):
            return self.simmap[hkey]
        else:
            print "!!!"
            return 0
    def gethash(self,genei,genej):
        if genei<genej:
            return str(genei)+":"+str(genej)
        else:
            return str(genej)+":"+str(genei)

# return the value of gene that is in wemodel, otherwise return None
def isinmodel(gene,wemodel):
    if gene in wemodel.vocab:
        return gene
    elif gene.lower() in wemodel.vocab:
        return gene.lower()
    elif gene.upper() in wemodel.vocab:
        return gene.upper()
    else:
        return None



def checkgenecoverage(genelist,wemodel):
    # checks the number of genes are in the model
    numgenes=len(genelist)
    numingenes=0.0
    for i in range(0,numgenes):
        genei = isinmodel(genelist[i],wemodel) 
        if (genei is not None):
            numingenes=numingenes+1
    print "num genes "+str(numgenes)+", num found genes "+str(numingenes) + ", percent of found genes "+str(100.0*(numingenes/numgenes))

def computesemanticrelatedness(intact,genelist,wemodel):
    # sets intact[i,j,INTACT_SR_INDEX] to the semantic relatedness of genes, range [0,1]  
    numgenes=len(genelist)
    for i in range(0,numgenes):
        for j in range(i,numgenes):    
            if wemodel==None:
                intact[i,j,INTACT_SR_INDEX]=0 #random.uniform(0, 1)
            else:
                genei = isinmodel(genelist[i],wemodel) 
                if (genei is None):
                    #print "Gene not found in embedded model "+str(genelist[i])                    
                    random.uniform(0, 1)                 
                else:
                    genej = isinmodel(genelist[j],wemodel)
                    if (genej is None):
                        #print "Gene not found in embedded model "+str(genelist[j])                      
                        random.uniform(0, 1)                 
                    else:       
                        #print "Gene pair found in embedded model "+str(genelist[i])   +","+str(genelist[j])+", sim "+str(wemodel.similarity(genei, genej))
                        intact[i,j,INTACT_SR_INDEX]=wemodel.similarity(genei, genej)
            intact[j,i,INTACT_SR_INDEX]=intact[i,j,INTACT_SR_INDEX]


def computeknownrelations(intgraph,geneindexmap,dfpairlist,evaldate):
    # sets intact[i,j,INTACT_KR_INDEX] to the relatedness of genes based on known interactions that occur before evaldate
    # range [-1,1]
    # the interaction score will be negative if the relations are known negative, positive if known positive
    #   the magnitude of the score will increase with the evidence
    #
    intgraph[:,:,INTACT_KR_INDEX]=0
    for row in dfpairlist.iterrows():
        genea = row[1]['GeneA']
        geneb = row[1]['GeneB']      
        intdate = row[1]['UTCDate']
        if intdate>evaldate:
            continue

        geneaindex=geneindexmap[genea]
        genebindex=geneindexmap[geneb]    
        if row[1]['Negative'] is True:
            intgraph[geneaindex,genebindex,INTACT_KR_INDEX]=intgraph[geneaindex,genebindex,INTACT_KR_INDEX]-0.5
            intgraph[geneaindex,genebindex,INTACT_KR_INDEX]=max(-1.0,intgraph[geneaindex,genebindex,INTACT_KR_INDEX])
        else:
            intgraph[geneaindex,genebindex,INTACT_KR_INDEX]=intgraph[geneaindex,genebindex,INTACT_KR_INDEX]+0.5
            intgraph[geneaindex,genebindex,INTACT_KR_INDEX]=min(1.0,intgraph[geneaindex,genebindex,INTACT_KR_INDEX])

        
def computeknownfirstdate(intgraph,geneindexmap,dfpairlist):
    # sets intact[i,j,INTACT_FD_INDEX] to the first date of evidence for the known relationship, range utc date or -1 if none
    intgraph[:,:,INTACT_FD_INDEX]=-1
    for row in dfpairlist.iterrows():
        genea = row[1]['GeneA']
        geneb = row[1]['GeneB']
        geneaindex=geneindexmap[genea]
        genebindex=geneindexmap[geneb]        
        
        intdate = row[1]['UTCDate']
        if intgraph[geneaindex,genebindex,INTACT_FD_INDEX]==-1 or intdate<intgraph[geneaindex,genebindex,INTACT_FD_INDEX]:
            intgraph[geneaindex,genebindex,INTACT_FD_INDEX]=intdate


def computepredrelations_old(intgraph,genelist,geneindexmap,targetgene):
    # compute predictions of all genes with the target gene   
    #   make predictions as to whether genes in genelist interact with the target
    #   gene based on the semantic relation of genes in intgraph
    #   Approach
    #       for each gene in genelist
    #           look at number of neighbors Ns with strong semantic relatedness
    #           identify the number of genes Ni in Ns that have interactions with the gene-gene pair of interest      
    #           record the values Ni, Ni/Ns
    #   returns a data frame with columns gene,ns,ni,ni/ns
    numgenes=len(genelist)
    Tns=0.2     # threshold for determining Ns
    Tng=0.2     # threshold for determining Ni
    targetgeneindex=geneindexmap[targetgene]
    rows_list = []
    for row in dfpairlist.iterrows():
        intdate = row[1]['UTCDate']
        if intgraph[geneaindex,genebindex,INTACT_FD_INDEX]==-1 or intdate<intgraph[geneaindex,genebindex,INTACT_FD_INDEX]:
            intgraph[geneaindex,genebindex,INTACT_FD_INDEX]=intdate
    for curgene in genelist:
        curgeneindex=geneindexmap[curgene]
        ns=0.0
        ni=0.0
        for neighborgeneindex in range(0,numgenes):
            if intgraph[curgeneindex,neighborgeneindex,INTACT_SR_INDEX]>Tns:
                ns=ns+1
                if intgraph[neighborgeneindex,targetgeneindex,INTACT_KR_INDEX]>Tng: 
                    ni=ni+1
                    
        if ns==0:
            nidivns=-1
        else:
            nidivns=(ni/ns)            
        dict1 = {'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Ni/Ns' : nidivns, 'Hit' : False, 'INTACT_FD_INDEX' : str(intdate) + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(intdate)))}
        rows_list.append(dict1)  
    return pd.DataFrame(rows_list)


def computepredrelations(intgraph,genelist,geneindexmap,targetgene):
    # compute predictions of all genes with the target gene   
    #       for a given gene, check if the gene has strong semantic relations with genes known to interact with the target gene
    numgenes=len(genelist)
    Tns=0.7     # threshold for determining Ns
    Tng=0.4     # threshold for determining Ni
    targetgeneindex=geneindexmap[targetgene]
    rows_list = []
    
    for curgene in genelist:
        curgeneindex=geneindexmap[curgene]
        ns=0.0
        ni=0.0
        for neighborgeneindex in range(0,numgenes):
            if intgraph[curgeneindex,neighborgeneindex,INTACT_SR_INDEX]>Tns:
                ns=ns+1
                if intgraph[neighborgeneindex,targetgeneindex,INTACT_KR_INDEX]>Tng: 
                    ni=ni+1
                    
        if ns==0:
            nidivns=-1
        else:
            nidivns=(ni/ns)            
        dict1 = {'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Score' : nidivns}
        rows_list.append(dict1)  
    return pd.DataFrame(rows_list)



def evaluatepredinteractions(intgraph,dfresults,evaldate,targetgene):
    # given a prediction in dfresults, check and see if the relation already exists
    # before the evaldate in which case mark it as a non hit, otherwise its a hit
    results=[]
    for row in dfresults.iterrows():
        genea = row[1]['Gene']
        score = row[1]['Score']
        geneaindex=geneindexmap[genea]
        genebindex=geneindexmap[targetgene]
        firstdate=intgraph[geneaindex,genebindex,INTACT_FD_INDEX]
        if score>0 and (firstdate==-1 or firstdate>=evaldate):
            results.append(True)
        else:
            results.append(False)
    dfresults['Hit']=results
            
##################################
##################################

debug=True

inFile = "/projects/mipseq/chemotext2/IntActKinaseP53Pairs.csv"
# FOR DEBUGGING inFile = "C:\Users\cpschmitt\Desktop\development\kinases\TestSet1_IntActPairs.csv"
# inWEFile = "/projects/chemotext/kinase_analysis/wordembedding_models/examplemodel"
inWEFile = "/projects/stars/var/chemotext/w2v/gensim/cumulative/pmc-2016.w2v"
outPredResultsFile = "/projects/mipseq/chemotext2/KinaseP53Resultstest_Comulative2016.csv"

intgraph=None                  # main array, [i][j]=ith gene by jth gene
INTACT_SR_INDEX=0              # [i][j][0]=semantic similarity of gene i and gene j (0,1)
INTACT_KR_INDEX=1              # [i][j][1]=interaction score of gene i and gene j from known sources (-1 to 1)
INTACT_FD_INDEX=2              # [i][j][2]=first date of gene i and gene j from known sources (utc, -1 if unknown)
INTACT_NUM_DIMENSIONS=3        # number of dimensions of 

# get evaluation date (date of word embedding generation)
evaldatestr = "01/01/2016" # jan 1 2012 0:0:0
#evaldate=1356998400  # jan 1 2013 0:0:0
evaldate = time.mktime(datetime.datetime.strptime(evaldatestr, "%d/%m/%Y").timetuple())

# target gene
targetgene="TP53"

# step 1 read in gene interaction data
dfpairlist=pd.read_csv(inFile)

# step 2 get list of genes and a map from gene to index within the genelist, the map will be used
#   to go from gene name to index within genelist and intgraph
geneset=Set(dfpairlist['GeneA'].tolist())
geneset=geneset.union(dfpairlist['GeneB'].tolist())
genelist=sorted(geneset)
geneindexmap = {x:i for i,x in enumerate(genelist)}

# construct intact 
numgenes=len(genelist)
intact=np.zeros((numgenes,numgenes,INTACT_NUM_DIMENSIONS))

# step 3 read in the word embeddings model
wemodel = gensim.models.Word2Vec.load(inWEFile) 
checkgenecoverage(genelist,wemodel)
# FOR DEBUGGING wemodel= testmodel("C:\Users\cpschmitt\Desktop\development\kinases\TestSet1_IntActSemRep.csv")

# step 4 update intact with semantic relatedness between genes
computesemanticrelatedness(intact,genelist,wemodel)
#print intact[geneindexmap['K1'],geneindexmap['P1'],INTACT_SR_INDEX]
#print intact[geneindexmap['P2'],geneindexmap['K1'],INTACT_SR_INDEX]
#print intact[geneindexmap['P1'],geneindexmap['K1'],INTACT_SR_INDEX]

# compute known gene-gene interaction data
computeknownrelations(intact,geneindexmap,dfpairlist,evaldate)
#print intact[geneindexmap['P1'],geneindexmap['TP53'],INTACT_KR_INDEX]
#print intact[geneindexmap['P2'],geneindexmap['TP53'],INTACT_KR_INDEX]
#print intact[geneindexmap['P10'],geneindexmap['TP53'],INTACT_KR_INDEX]
#print intact[geneindexmap['K1'],geneindexmap['TP53'],INTACT_KR_INDEX]
#print intact[geneindexmap['K1'],geneindexmap['Tmp1'],INTACT_KR_INDEX]
#print intact[geneindexmap['K1'],geneindexmap['K2'],INTACT_KR_INDEX]

# get date of first known relations
computeknownfirstdate(intact,geneindexmap,dfpairlist)
#print intact[geneindexmap['P1'],geneindexmap['TP53'],INTACT_FD_INDEX]
#print intact[geneindexmap['P5'],geneindexmap['TP53'],INTACT_FD_INDEX]
#print intact[geneindexmap['K1'],geneindexmap['TP53'],INTACT_FD_INDEX] 
#print intact[geneindexmap['K2'],geneindexmap['TP53'],INTACT_FD_INDEX] 
  
# predict novel gene-gene interactions based upon semantic relatedness graph
dfresults=computepredrelations(intact,genelist,geneindexmap,targetgene)
    
# evaluate predictions
evaluatepredinteractions(intact,dfresults,evaldate,targetgene)

dfresults.to_csv(outPredResultsFile)
