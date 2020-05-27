CMD: Change directory path to the folder that contains these files:
    * main_LR_hypertun.py
    * README_JJ_LR.txt

There are 11 optional arguments:
# Parsing Optional arguments
    -uz , --unzip_IMDB, help="Unzips the IMDB tar.gz file (string)."             , choices=['unzip', 'no']                                   , default='no'
    -ngn, --n_gram_min, help="Minimum number of words to group (integer)."       , type=int                                                  , default=1
    -ngx, --n_gram_max, help="Maximum number of words to group (integer)."       , type=int                                                  , default=1   
    -mi , --max_iter  , help="LR maximum iteration (integer)."                   , type=int                                                  , default=1000
    -sv , --solver    , help="LR Solver algorithm (string)."                     , choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], default='lbfgs'
    -rs , --rand_seed , help="Random Number Generator Seed (integer)."           , type=int                                                  , default=42      
    -cv , --cross_val , help="Cross Validation iterations (integer)."            , type=int                                                  , default=5
    -j  , --jobs      , help="Number of processors used (integer)."              , type=int,            choices=range(-1,9)                  , default=-1
    -trz, --train_size, help="Starting IMDB row index (integer)."                , type=int                                                  , default=0   
    -tsz, --test_size , help="Decision Tree Depth Range minimum (integer)."      , type=int                                                  , default=0    
    -sp , --split     , help="The split percentage used (0.0 < x < 1.0) (float).", type=float                                                , default=0.2

Sample CMD input:

python main_LR_hypertun.py
python main_LR_hypertun.py -trz 5000
python main_LR_hypertun.py -trz 5000 -tsz 500
python main_LR_hypertun.py -trz 5000 -sp 0.3
python main_LR_hypertun.py -uz unzip
python main_LR_hypertun.py -ngn 2 -ngx 3
python main_LR_hypertun.py -mi 7600
python main_LR_hypertun.py -sv saga
python main_LR_hypertun.py -cv 10
python main_LR_hypertun.py -j 4

python main_LR_IMDB.py -trz 5000 -sp 0.35 -uz no -ngn 1 -ngx 2 -mi 1000 -sv saga -cv 10 -j 8
