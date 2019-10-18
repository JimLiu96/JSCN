MODEL = 'SpectralCF'
DATASET = 'amazon'

EMB_DIM = 32
BATCH_SIZE = 1024
DECAY = 0.001
LAMDA = 1
K = 5
N_EPOCH = 20000
LR = 0.001
DROPOUT = 0.0

DIR = './data/'+DATASET+'/'
# metaName = 'ratings_Tools_and_Home_Improvement'
# metaName = 'ratings_Amazon_Instant_Video'
# metaName = 'ratings_Home_and_Kitchen'
# # # metaName = 'ratings_Digital_Music'

# userEmbeddingDIR = './embedding/' + metaName + '_userEmbedding.pickle'
# itemEmbeddingDIR = './embedding/' + metaName + '_itemEmbedding.pickle'
# trainUserFileName = metaName + '_train_user.dat'
# testUserFileName = metaName + '_test_user.dat'
# eigenPickleFile = DIR + metaName + '_eigenPickle.pickle'


# metaName_1 = 'ratings_Books'
# metaName_2 = 'ratings_Movies_and_TV'
# metaName_1 = 'ratings_Tools_and_Home_Improvement'
# metaName_2 = 'ratings_Home_and_Kitchen'
metaName_1 = 'ratings_Amazon_Instant_Video'
# metaName_2 = 'ratings_Video_Games'
# metaName_2 = 'ratings_Movies_and_TV' + '_from_' + metaName_1
# metaName_2 = 'ratings_Books' + '_from_' + metaName_1
# metaName_2 = 'ratings_Electronics' + '_from_' + metaName_1
# metaName_2 = 'ratings_CDs_and_Vinyl' + '_from_' + metaName_1
# metaName_2 = 'ratings_Clothing_Shoes_and_Jewelry' + '_from_' + metaName_1
# metaName_2 = 'ratings_Home_and_Kitchen' + '_from_' + metaName_1
# metaName_2 = 'ratings_Kindle_Store' + '_from_' + metaName_1
# metaName_2 = 'ratings_Sports_and_Outdoors' + '_from_' + metaName_1
# metaName_2 = 'ratings_Cell_Phones_and_Accessories' + '_from_' + metaName_1
# metaName_2 = 'ratings_Health_and_Personal_Care' + '_from_' + metaName_1
# metaName_2 = 'ratings_Toys_and_Games' + '_from_' + metaName_1
# metaName_2 = 'ratings_Video_Games' + '_from_' + metaName_1
# metaName_2 = 'ratings_Tools_and_Home_Improvement' + '_from_' + metaName_1
# metaName_2 = 'ratings_Beauty' + '_from_' + metaName_1
metaName_2 = 'ratings_Apps_for_Android' + '_from_' + metaName_1
# metaName_2 = 'ratings_Office_Products' + '_from_' + metaName_1
# metaName_2 = 'ratings_Pet_Supplies' + '_from_' + metaName_1
# metaName_2 = 'ratings_Automotive' + '_from_' + metaName_1
# metaName_2 = 'ratings_Grocery_and_Gourmet_Food' + '_from_' + metaName_1
# metaName_2 = 'ratings_Patio_Lawn_and_Garden' + '_from_' + metaName_1
# metaName_2 = 'ratings_Baby' + '_from_' + metaName_1
# metaName_2 = 'ratings_Digital_Music' + '_from_' + metaName_1
# metaName_2 = 'ratings_Musical_Instruments' + '_from_' + metaName_1

metaName_3 = 'ratings_Health_and_Personal_Care' + '_from_' + metaName_1
# metaName_3 = 'ratings_Office_Products' + '_from_' + metaName_1

metaName_4 = 'ratings_Office_Products' + '_from_' + metaName_1


userEmbeddingDIR_1 = './embedding/' + metaName_1 + '_userEmbedding.pickle'
itemEmbeddingDIR_1 = './embedding/' + metaName_1 + '_itemEmbedding.pickle'
trainUserFileName_1 = metaName_1 + '_train_user.dat'
testUserFileName_1 = metaName_1 + '_test_user.dat'
eigenPickleFile_1 = DIR + metaName_1 + '_eigenPickle.pickle'

userEmbeddingDIR_2 = './embedding/' + metaName_2 + '_userEmbedding.pickle'
itemEmbeddingDIR_2 = './embedding/' + metaName_2 + '_itemEmbedding.pickle'
trainUserFileName_2 = metaName_2 + '_train_user.dat'
testUserFileName_2 = metaName_2 + '_test_user.dat'
eigenPickleFile_2 = DIR + metaName_2 + '_eigenPickle.pickle'

# userEmbeddingDIR_3 = './embedding/' + metaName_3 + '_userEmbedding.pickle'
# itemEmbeddingDIR_3 = './embedding/' + metaName_3 + '_itemEmbedding.pickle'
# trainUserFileName_3 = metaName_3 + '_train_user.dat'
# testUserFileName_3 = metaName_3 + '_test_user.dat'
# eigenPickleFile_3 = DIR + metaName_3 + '_eigenPickle.pickle'

# userEmbeddingDIR_4 = './embedding/' + metaName_4 + '_userEmbedding.pickle'
# itemEmbeddingDIR_4 = './embedding/' + metaName_4 + '_itemEmbedding.pickle'
# trainUserFileName_4 = metaName_4 + '_train_user.dat'
# testUserFileName_4 = metaName_4 + '_test_user.dat'
# eigenPickleFile_4 = DIR + metaName_4 + '_eigenPickle.pickle'
commonUserFileName = DIR +  metaName_1 + '-' + metaName_2 + '-' + 'commonUserIDpair.pickle'
# commonUserFileName_12 = DIR +  metaName_1 + '-' + metaName_2 + '-' + 'commonUserIDpair.pickle'
# commonUserFileName_13 = DIR +  metaName_1 + '-' + metaName_3 + '-' + 'commonUserIDpair.pickle'
# commonUserFileName_14 = DIR +  metaName_1 + '-' + metaName_4 + '-' + 'commonUserIDpair.pickle'

CKPTFILENAME = ''
