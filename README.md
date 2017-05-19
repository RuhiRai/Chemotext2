# M2M
Code to use semantic relatedness of small molecules to predict new 
molecule to molecule relationships

Input file:
IntActKinaseP53Pairs.csv ---- interactions between genea and geneb with pubmedIDs of articles from where the interactions are obtatined.

AllkinaseInDetail620.csv has 620 kinases downloaded from http://kinase.com/human/kinome/ --- The file contains  Gene IDs, refseq accessions, HGNC names and improved sequences for some kinases.

AllKinase620.csv file contains names of the kinases in lower case. Kinase names are obtained from AllkinaseInDetail620.csv
model2011_2yr

Scripts:
moleculeSemanticRelatedness.py ---- outputs {'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Ni/Ns' : (ni/ns), 'Hit' : true/false}
where hit is true if no existing known relationship exists between the gene
and the target gene for a given evaluation date

eimdatagenerator.v8.py ----- Code to generate data for training the EIM entity-entity relationship model.  The data
consists of a matrix where each row represents a ei-ej pair and the columns hold the
sikj value (along with ei and ej indices, known relationship date, case,...)

eimmodel.py: Module for building a baseline model for predicting entity entity interactions from eimdatagenerator

eimsplit.py: creates test and training set
