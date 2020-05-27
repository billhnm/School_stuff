## Status

Final Stage 4 report posted: Project Report for Stage 4-v3.docx

Files rearranged into sub-directories for ease of review:
* Datasets: 1250 and 5k IMDB subsets, IMDB stop words, dataset description, full dataset URL
* Code: python code for preprocessing, existing classifiers (Logistic Regression, Multinomial Naive Bayes, Multi-layer Perceptron, SVM), etc.
* Charts_results: Initial charts, result summaries, etc. 
* Batch_instructions_files: batch and readme files for running code

Initial SVM classifier complete
* Test accuracy/f1: 0.84/0.84
* IDFSVM/main.py file posted with associated modules in /modules subdirectory

Initial Logistic Regression classifier complete
* Test accuracy/f1: 0.866/0.866
* project_LR.py file posted

Initial Multi-Layer Perceptron Neural Net Model Results based on 5k dataset
* Train Accuracy: 0.980, Test Accuracy: 0.855
* Train f1: 0.980, Test f1: 0.855
* ROC_AUC: 0.855
* updated project_mlp.py file posted
* Project_mlp_neural_network_results.docx file posted with intermediate and alternate results

Naive Bayes results on 1250 dataset, adding stop words and bigrams
* Train Accuracy: 0.990, Test Accuracy: 0.843
* Train f1: 0.990, Test f1: 0.833
* ROC_AUC: 0.844
* updated project_mnb.py file posted
* Project_mnb_results.docx file posted with intermediate and alternate results
* mnb_roc_curve.png posted
* imdb_stop_words.data file added based on initial analysis

