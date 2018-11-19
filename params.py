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
# # metaName = 'ratings_Home_and_Kitchen'
# # metaName = 'ratings_Digital_Music'

# userEmbeddingDIR = './embedding/' + metaName + '_userEmbedding.pickle'
# itemEmbeddingDIR = './embedding/' + metaName + '_itemEmbedding.pickle'
# trainUserFileName = metaName + '_train_user.dat'
# testUserFileName = metaName + '_test_user.dat'
# eigenPickleFile = DIR + metaName + '_eigenPickle.pickle'


# metaName_1 = 'ratings_Video_Games'
# metaName_2 = 'ratings_Home_and_Kitchen'
metaName_1 = 'ratings_Amazon_Instant_Video'
metaName_2 = 'ratings_Automotive'
# metaName_2 = 'ratings_Digital_Music'
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
commonUserFileName = DIR +  metaName_1 + '-' + metaName_2 + '-' + 'commonUserIDpair.pickle'

CKPTFILENAME = ''
