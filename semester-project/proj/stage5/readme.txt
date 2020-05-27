Los Ancianos
Program flow: 

To be able to handle the data and multiple algorithms, our program works in three stages:

Stage 1: Read and pre-process the data -- we read the data and apply the necessary preprocessing for each classifier.
	- in each case, the preprocessing program calls the respective code module (project_LR.py, project_mnb.py, ... etc.) 
        - once that is complete, each classifier's preprocessing steps writes their pre-processed word vectors to disk. 

python project_preproc.py lr
python project_preproc.py mlp
python project_preproc.py mnb
python project_preproc.py rf
python project_preproc.py svm

Stage 2: Read the pre-processed word vectors in from disk and run the appropriate classifier
	- as above, the classifier call the respective code module.
        - once that is complete, each classifier writes its predictions to disk.

python project_classify.py lr
python project_classify.py mlp
python project_classify.py mnb
python project_classify.py rf
python project_classify.py svm

Stage 3: Read in the individual classifier predictions and then determine final prediction by majority vote.

python project_classify.py vote 

=====

Dataset information: 

The analysis used the IMDB "Large Movie Review Dataset", v1.0.  This contains 50,000 reviews, coded as either 'pos' (positive) or 'neg' (negative).  The dataset is balanced between positive and negative reviews.  The dataset is already split evenly between 25k test and training subsets.  The authors made sure that no movie had more than 30 reviews, to keep popular movies from skewing the results.  Also, the set of movies reviewed in the test and training datasets is different.  

The reviews were classified as either positive or negative based on IMDB score for the review.  Negative reviews received ratings of <= 4 out of 10, and positive reviews received ratings of >= 7 out of 10.  

The only preprocessing of the dataset itself was to switch the ratings for 'pos' and 'neg' to (1, -1).  

Dataset URL: 
http://ai.stanford.edu/~amaas/data/sentiment/ 