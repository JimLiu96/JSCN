This is a code for cross-domain recommendation. It supports using multiple source domains to improve the performance on on one target domain. The code is based [spectral collaborative filtering](https://github.com/lzheng21/SpectralCF) method.

# Citation
If you use the code, please cite [our paper](https://arxiv.org/abs/1910.08219)
```
@INPROCEEDINGS{9006266,
  author={Z. {Liu} and L. {Zheng} and J. {Zhang} and J. {Han} and P. S. {Yu}},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)},
  title={JSCN: Joint Spectral Convolutional Network for Cross Domain Recommendation},
  year={2019},
  pages={850-859},}
```

# JSCN-cross-domain
Joint spectral convolutional network for cross domain recommendation with 

We present how to run the code for JSCN-beta with sigle source domain and multiple source domains as JSCN_beta_s1.py, JSCN_beta_s2.py respectively. 

The data is from two domains:\
**The target domain**: Amazon Instant Video\
**The source domain**: Apps for Android

## Environment
```
Tensorflow = 1.4.1
Python = 3.6
```

## Run 
``$ python run.py``

- It may take a few minutes to compute the eigenvectors at the first time of computation. Then the eigenvectors are saved locally and do not require computation later.

- After 200 epoch, the model will be evaluated by testing the MAP and Recall

## Change to new data
There are several important part you may need to change:
- `params.py`: the metaName-1 is the target domain file name, the metaName-2,3,4,... is the source domain file name. The format of the data can be found in data section. The `commonUserFileName` denotes the alignment of users in target domain and source domain. `commonUserFileName_12` means the alignment between metaName-1 and meta-Name-2.
- `commonuser_file.pickle`, Generate the common user alignment pickle list

# Data
We use the `Amazon_rating_data_set`, which can be downloaded [here](http://jmcauley.ucsd.edu/data/amazon/).

The processing file is `./data/amazon/preprocess.py` and the using cases are in `./data/amazon/dataPreprocessing.ipynb`.

To use for new datasets, you may need to create a cross domain training data.

## Data format
- The rating files: `SourcedomainRating.txt`, `targetdomainRating.txt`, the name should be change according to the name in the `params.py` file:\
Each row is user_id, basket_id, basket_id, ... 
  
- The commonuser files: `commonuser.pickle`:
  It is a python tuple list : 
  ```[(uid_in_target_1, uid_in_source_1),(uid_in_target_2, uid_in_source_2),....,]```
  which tells the model how the users in target domain are aligned with users in source domains.

You should generate these files first if you want to run the code on new datasets.

The maximum number of source domains is currently 2 by using `JSCN_beta_s2`
