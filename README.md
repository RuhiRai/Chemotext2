# M2M
Code to use semantic relatedness of small molecules to predict new 
molecule to molecule relationships

Input file:
IntActKinaseP53Pairs.csv ---- Biological interactions file with pubmedIDs

Scripts:
moleculeSemanticRelatedness.py ---- outputs {'Gene': curgene, 'Ni': ni, 'Ns': ns, 'Ni/Ns' : (ni/ns), 'Hit' : true/false}
where hit is true if no existing known relationship exists between the gene
and the target gene for a given evaluation date

eimdatagenerator.v8.py ----- Code to generate data for training the EIM entity-entity relationship model.  The data
consists of a matrix where each row represents a ei-ej pair and the columns hold the
sikj value (along with ei and ej indices, known relationship date, case,...)
