"""
Created on Thu Nov 03 16:29:29 2016

Module for building a baseline model for predicting entity 
entity interactions from eimdatagenerator

Each row of input data set contains an ei, ej pair and the values for all eijk
In addition each row holds the indicated relationship date between ei and ej
and a Case indicator where

original case (origcase):
0=relationship date < evaluation date
1=relationship date >= evaluation date
2=no relationship date

recode case to (modelcase):
0=0
1,2=1

evaluation: two measures of interest

- accuracy: accuracy in predicting model case 0 or 1.  This is less interesting, but is the
criteria for model building

- forecast: ability to pick origcase=1 in the top list of modelcases which are predicted to be 1

  - np = # of original cases that are 1
  - nc = # of predicated cases equal to 1 that are actually 1 in the original case
  - nc20 = # of predicated cases equal to 1 that are actually 1 in the original case and 
           are in the top 20 cases predicted to be a 1

  nc20 is the evaluation measure for forecast

  
  FIX - hypergeometric test requires the actual numbers prior to filtering!!!!!!!!!!!!!!

"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
import gensim.models
import scipy.stats as st
from datetime import date
import sys



# get training data
#fmodeldata="modelv8_e20131_p1-12_d24.txt"
fmodeldata = (sys.argv[1])

ftraindata = fmodeldata+".train.txt"
ftestdata = fmodeldata+".test.txt"

dftrain=pd.read_csv(ftraindata,na_values=["None","nan"])
dftrain.rename(columns={'case': 'origcase'}, inplace=True)
dftrain['predcase'] = dftrain['origcase'].replace(['1', '2'], 1)
dftrain.fillna(0,inplace=True)
xtrain=dftrain.drop(['entitya', 'entityb', 'relposixdate', 'reldate', 'metimediff', 'predcase', 'origcase'], axis=1)
ytrain=dftrain['predcase']

# scale if necessary 
#   not needed for SGD


# grid search on hyperparameters (alpha for SGD)
#    estimator (regressor/classifier)
#    parameter space
#    search method
#    cross-validation scheme
#    scoring function  (SGDClassifier provides a score function which is the mean accuracy on test data and label.
#

numcvs=3
param_grid=[{'alpha':[0.0001,0.00001],'penalty':['l2'],
             'fit_intercept':[True],'loss':['modified_huber'],'n_iter':[15]}]

clf = GridSearchCV(SGDClassifier(), param_grid, cv=numcvs)
#param_grid=[{'strategy':["uniform"]}]
#clf = GridSearchCV(DummyClassifier(), param_grid, cv=numcvs)


clf.fit(xtrain, ytrain)

fresultsname = fmodeldata+".sgd.txt"
fresults=open(fresultsname,"w")
fresults.write("Baseline Training Results\n")
fresults.write("GridSearchCV; cv="+str(numcvs)+", grid="+str(param_grid)+"\n")
fresults.write("Best score "+str(clf.best_score_)+"\n")
fresults.write("Best estimator "+str(clf.best_estimator_)+"\n")
fresults.write("Best parameters "+str(clf.best_params_)+"\n")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
     fresults.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

# test, remember to scale by same scaling as train data if scaling was used
fresults.write("\nSGDClassifier Testing Results\n")
dftest=pd.read_csv(ftestdata,na_values=["None","nan"])
dftest.rename(columns={'case': 'origcase'}, inplace=True)
dftest['predcase'] = dftest['origcase'].replace(['1', '2'], 1)
dftest.fillna(0,inplace=True)
xtest=dftest.drop(['entitya', 'entityb', 'relposixdate', 'reldate', 'metimediff', 'predcase', 'origcase'], axis=1)
ytest=dftest['predcase']


ytrue, ypred = ytest, clf.predict(xtest)
#res=classification_report(ytrue, ypred, target_names=target_names)
res=classification_report(ytrue, ypred)
fresults.write(res+"\n")

predprob=clf.predict_proba(xtest)

# now, evaluate on the real test
#   first, get the predicted log probabilites
dfpred=pd.DataFrame.from_records(predprob, columns=["0","1"])
dftest['pred0']=dfpred['0']
dftest['pred1']=dfpred['1']

# now get rid of known relationships and sort by predicted relationship
dftestuk=dftest[dftest['origcase']!=0]
dftestuk.sort_values("pred0",ascending=False,inplace=True)
numunknowns=len(dftestuk)
numcase1=len(dftest[dftest['origcase']==1])
numcase2=len(dftest[dftest['origcase']==2])

#dfhits=dftestuk[(dftestuk['origcase'] == 1) &  (dftestuk['predcase'] == 1)]
#numhits=len(dfhits)

# now get number of predicted relations in the top 20 and print out those relationships
dfn20=dftestuk.head(20)
dfn20hits=dfn20[(dfn20['origcase'] == 1) &  (dfn20['predcase'] == 1)]
numtop20hits=len(dfn20hits)

hyper=st.hypergeom.sf(numtop20hits,numunknowns,numcase1,20)
fresults.write("NC20 "+str(numtop20hits)+", num "+str(numunknowns)+", num possible "+str(numcase1)+", hyper "+str(hyper)+"\n\n")
fresults.close()
dfn20hits.to_csv(fresultsname,  mode='a')

# now get number of predicted relations in the top 20 fpr TP53 and print out those relationships
dftestuktp53=dftestuk[(dftestuk['entitya']=='tp53') | (dftestuk['entityb']=='tp53')]
dftestuktp53.sort_values("pred0",ascending=False,inplace=True)
numunknownstp53=len(dftestuktp53)
numcase1tp53=len(dftestuktp53[dftestuktp53['origcase']==1])

dfn20tp53=dftestuktp53.head(20)
dfn20hitstp53=dfn20tp53[(dfn20tp53['origcase'] == 1) &  (dfn20tp53['predcase'] == 1)]
numtop20hitstp53=len(dfn20hitstp53)
hyper=st.hypergeom.sf(numtop20hitstp53,numunknownstp53,numcase1tp53,20)
fresults=open(fresultsname,"a")
fresults.write("NC20 "+str(numtop20hitstp53)+", num "+str(numunknownstp53)+", num possible "+str(numcase1tp53)+", hyper "+str(hyper)+"\n\n")
fresults.close()
#dftestuktp53.to_csv(fresultsname,  mode='a')
dfn20hitstp53.to_csv(fresultsname,  mode='a')

