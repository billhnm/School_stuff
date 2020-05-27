usage: main.py [-h] -k KERNEL -i INFILE [-g GAMMA]

Analysis of dimenstionality reduction techniques for a given dataset
(specified as a dataset from scikit-learn (-i and -s))

optional arguments:
  -h, --help            show this help message and exit
  -k KERNEL, --kernel KERNEL
                        SVM Kernel: linear or rbf
  -i INFILE, --input INFILE
                        Input Dataset: filename...
  -g GAMMA, --gamma GAMMA
                        SVM Tunable for rbf kernel. Default:0 -- OPTIONAL


Example - SVM RBF Kernel w/ gamma 1.0
  python main.py -k rbf -i ..\IMDB_1250.data -g 1.0

Example - SVM linear Kernel
  python main.py -k linear -i ..\IMDB_1250.data
