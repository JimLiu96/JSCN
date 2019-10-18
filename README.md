# JSCN-cross-domain
Joint spectral convolutional network for cross domain recommendation

This is a repository create for paper reviewing. We present how to run the code for JSCN-beta with sigle source domain and multiple source domains as JSCN_beta_s1.py, JSCN_beta_s2.py respectively.

The data is from two domains:\
**The target domain**: Amazon Instant Video\
**The source domain**: Apps for Android

Tensorflow = 1.4.1
Python = 3.6

run `$ python run.py`

- It may take a few minutes to compute the eigenvectors at the first time of computation. Then the eigenvectors are saved locally and do not require computation later.

- After 200 epoch, the model will be evaluated by testing the MAP and Recall
