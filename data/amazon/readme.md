## datasets
All the datasets can be downloaded from http://jmcauley.ucsd.edu/data/amazon/


## preprocess.py
```python
domainMetaName = 'ratings_Amazon_Instant_Video'
ratingFileName = domainMetaName+'.csv' 
preprocess.getAllUserItems(ratingFileName) # get all the userID and itemID
```

## Notebook

``dataProcessing.ipynb`` contains the code for how to use ``preprocess.py`` to process the raw data

## Directory
You need to shift the ``Path`` accordingly with your environment.
