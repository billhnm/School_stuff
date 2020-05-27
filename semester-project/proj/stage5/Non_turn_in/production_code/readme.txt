Los Ancianos
Program flow: 

To be able to handle the data and multiple algorithms, our program works in three stages:

Stage 1: Read and pre-process the data -- we read the data and apply the necessary preprocessing for each classifier.
         - once that is complete, each classifier's preprocessing steps writes their pre-processed word vectors to disk. 

python project_preproc.py lr
python project_preproc.py mlp
python project_preproc.py mnb
python project_preproc.py rf
python project_preproc.py svm

Stage 2: Read the pre-processed word vectors in from disk and run the appropriate classifier
         - once that is complete, each classifier writes its predictions to disk.

python project_classify.py lr
python project_classify.py mlp
python project_classify.py mnb
python project_classify.py rf
python project_classify.py svm

Stage 3: Read in the individual classifier predictions and then determine final prediction by majority vote.

python project_classify.py vote 