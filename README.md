# M2M
Code to use semantic relatedness of small molecules to predict new 
molecule to molecule relationships

Input file:
IntActKinaseP53Pairs.csv ---- interactions between genea and geneb with pubmedIDs of articles from where the interactions are obtatined.

AllkinaseInDetail620.csv has 620 kinases downloaded from http://kinase.com/human/kinome/ --- The file contains  Gene IDs, refseq accessions, HGNC names and improved sequences for some kinases.

AllKinase620.csv file contains names of the kinases in lower case. Kinase names are obtained from AllkinaseInDetail620.csv
model2011_2yr

top20kinaseSeq2011.fasta includes top 27 kinase from 2011 that had a hit using the sematic similarity algorithm. we wanted to perform a sequence similarity on the same set of genes to compare the distance matrix.

Sim_matrixtop20genes2011.csv has sequence similarity scores for top 27 kinases from 2011.

620KinaseSeq.fasta includes protein sequences for all 620 known kinases. The data is obtatined from http://kinase.com/human/kinome/ in particular from Kincat_Hsap.08.02.xls file.

620ProteinSeqSimScore.csv has all the 620 kinase sequence similarity score.


Scripts:
moleculeSemanticRelatedness.py ---- outputs {'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Ni/Ns' : (ni/ns), 'Hit' : true/false}
where hit is true if no existing known relationship exists between the gene
and the target gene for a given evaluation date

eimdatagenerator.v8.py ----- Code to generate data for training the EIM entity-entity relationship model.  The data
consists of a matrix where each row represents a ei-ej pair and the columns hold the
sikj value (along with ei and ej indices, known relationship date, case,...)

eimmodel.py: Module for building a baseline model for predicting entity entity interactions from eimdatagenerator

eimsplit.py: creates test and training set
