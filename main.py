from crossSpectralCF import *
# from test import *
import test
import tensorflow as tf
import pickle
import logging
# from params import *
import load_data
import params
import os

# Gloable Variable get from the test file: (Ignore)
# USER_NUM, ITEM_NUM, N_EPOCH
# data_generator.R: it is the relationship matrix of user and item (bi-graph)
logFileName = 'training.log'
with open(logFileName, 'a') as logFileA:
    logFileA.write(params.metaName_1 + '_' + params.metaName_2 + '\n')
logStrList = [0]*4
ckpt_dir = "./" + params.metaName_1 + '_' + params.metaName_2 + "_checkpoints/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
ckpt_log = 'ckptfile.log'

data_generator_1 = load_data.Data(train_file=params.DIR + params.trainUserFileName_1, test_file=params.DIR+params.testUserFileName_1,batch_size=params.BATCH_SIZE)
data_generator_2 = load_data.Data(train_file=params.DIR + params.trainUserFileName_2, test_file=params.DIR+params.testUserFileName_2,batch_size=params.BATCH_SIZE)
print(params.trainUserFileName_1, params.testUserFileName_1)
print(params.trainUserFileName_2, params.testUserFileName_2)
USER_NUM_1, ITEM_NUM_1 = data_generator_1.get_num_users_items()
USER_NUM_2, ITEM_NUM_2 = data_generator_2.get_num_users_items()

def modelTrain(testState = False):
    userEmbeddingFile_1 = open(params.userEmbeddingDIR_1 ,'wb')
    itemEmbeddingFile_1 = open(params.itemEmbeddingDIR_1 ,'wb')
    # graph_1 = data_generator_1.R
    # A_1 = adjacient_matrix(graph_1,self_connection=True)
    # D_1 = degree_matrix(A_1)
    # L_1 = laplacian_matrix(D_1,A_1,normalized=True)

    userEmbeddingFile_2 = open(params.userEmbeddingDIR_2 ,'wb')
    itemEmbeddingFile_2 = open(params.itemEmbeddingDIR_2 ,'wb')
    # graph_2 = data_generator_2.R
    # A_2 = adjacient_matrix(graph_2,self_connection=True)
    # D_2 = degree_matrix(A_2)
    # L_2 = laplacian_matrix(D_2,A_2,normalized=True)

    eigPara = []
    
    #### load or dump the eigen value
    print("load or dump the eigen value")
    eigenPickleFile_1 = params.eigenPickleFile_1
    eigenPickleFile_2 = params.eigenPickleFile_2
    if os.path.exists(eigenPickleFile_1):
        L_1 = 0
        graphLambda_1, graphU_1 = loadEigenvector(L_1, eigenPickleFile_1)
    else:
        graph_1 = data_generator_1.R
        A_1 = adjacient_matrix(graph_1,self_connection=True)
        D_1 = degree_matrix(A_1)
        L_1 = laplacian_matrix(D_1,A_1,normalized=True)
        graphLambda_1, graphU_1 = genEigenvector(L_1, eigenPickleFile_1)    
    if os.path.exists(eigenPickleFile_2):
        L_2 = 0
        graphLambda_2, graphU_2 = loadEigenvector(L_2, eigenPickleFile_2)
    else:
        graph_2 = data_generator_2.R
        A_2 = adjacient_matrix(graph_2,self_connection=True)
        D_2 = degree_matrix(A_2)
        L_2 = laplacian_matrix(D_2,A_2,normalized=True)
        graphLambda_2, graphU_2 = genEigenvector(L_2, eigenPickleFile_2)   
    graphLambda_1 = np.diag(graphLambda_1)
    graphLambda_feed_1 = graphLambda_1
    graphU_feed_1 = graphU_1
    graphLambda_2 = np.diag(graphLambda_2)
    graphLambda_feed_2 = graphLambda_2
    graphU_feed_2 = graphU_2

    paramsList = [params.EMB_DIM, params.BATCH_SIZE, params.DECAY, params.K, params.N_EPOCH, params.LR]
    # self, K, M = 1, n_users_1, n_users_2, n_items_1, n_items_2, emb_dim, lr, batch_size, decay, DIR
    model = SpectralCF(K=params.K, M=2, n_users_1=USER_NUM_1, n_users_2=USER_NUM_2, n_items_1=ITEM_NUM_1, n_items_2=ITEM_NUM_2,
                      emb_dim=params.EMB_DIM, lr=params.LR, decay=params.DECAY, batch_size=params.BATCH_SIZE,DIR=params.DIR)
    ## Configure of Tensorflow
    ckpt_name_meta = str(model.model_name + '_' + params.metaName_1 + 
                            '_' + str(params.metaName_2) + '_' + '_'.join([str(val) for val in paramsList]) + '.ckpt-')
#     ckptfileName = params.CKPTFILENAME
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    # sess = tf.Session(config=config)
        # graphLambda_feed = graphLambda
        # graphU_feed = graphU
        print("session created")
        sess.run(tf.global_variables_initializer())
        print("-------------------initialization finished-------------------")
        start = 0
        model_saver = tf.train.Saver(max_to_keep=10)
        if ckpt and ckpt.model_checkpoint_path:
            if ckpt.model_checkpoint_path.split("-")[0] == ckpt_name_meta:
                print(ckpt.model_checkpoint_path)
                start = int(ckpt.model_checkpoint_path.split("-")[1])
                logging.info("start by iteration: %d" % (start))
                model_saver = tf.train.Saver()
                model_saver.restore(sess, ckpt.model_checkpoint_path)
        try:
            for epoch in range(start, params.N_EPOCH):
                # if epoch == start:
                #     print(model.users)
        #         print("-----------Running Epoch %d-----------" % epoch)
                users_1, pos_items_1, neg_items_1 = data_generator_1.sample()
                users_2, pos_items_2, neg_items_2 = data_generator_2.sample()
                feed_dict = {    model.graphLambda_1: graphLambda_feed_1,
                                 model.U_1: graphU_feed_1,
                                 model.users_1: users_1,
                                 model.pos_items_1: pos_items_1, 
                                 model.neg_items_1: neg_items_1,
                                 model.graphLambda_2: graphLambda_feed_2,
                                 model.U_2: graphU_feed_2,
                                 model.users_2: users_2,
                                 model.pos_items_2: pos_items_2, 
                                 model.neg_items_2: neg_items_2,
                                }
                _, loss, u_embeddings_1, i_embeddings_1, u_embeddings_2, i_embeddings_2 = sess.run([model.updates, model.loss, 
                    model.user_embeddings_1, model.item_embeddings_1, model.user_embeddings_2, model.item_embeddings_2],
                    feed_dict=feed_dict)
                if epoch%1 == 0:
                    print(params.EMB_DIM,params.BATCH_SIZE,params.DECAY,params.K,params.N_EPOCH,params.LR)
                    print('Epoch %d training loss %f' % (epoch, loss))
                    if testState == True:
                        users_to_test = list(test.data_generator.test_set.keys())
                        ret = test.testAll(sess, model, users_to_test, graphLambda_feed, graphU_feed)
                        print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
                                % (ret[0],ret[1],ret[2],ret[3],ret[4]))
                        print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
                            % (ret[5], ret[6], ret[7], ret[8], ret[9]))
                        logStrList[0] = 'EMB_DIM,BATCH_SIZE,DECAY,K,N_EPOCH,LR:'+','.join([str(val) for val in paramsList])
                        logStrList[1] = 'Epoch ' + str(epoch) + 'training loss ' + str(loss)
                        logStrList[2] = 'recall_20_40_60_80:' + ','.join([str(val) for val in ret[:5]])
                        logStrList[3] = 'map_20_40_60_80_100:' + ','.join([str(val) for val in ret[5:]])
                        logStr = '\n'.join(logStrList)
                        with open(logFileName,'a') as logFileA:
                            logFileA.write(logStr)
                    else:
                        logStrList[0] = 'EMB_DIM,BATCH_SIZE,DECAY,K,N_EPOCH,LR:'+','.join([str(val) for val in paramsList])
                        logStrList[1] = 'Epoch ' + str(epoch) + 'training loss ' + str(loss)
                        logStrList[2] = ' '
                        logStrList[3] = ' '
                        logStr = '\n'.join(logStrList)
                        with open(logFileName,'a') as logFileA:
                            logFileA.write(logStr) 
                if epoch%int((USER_NUM_1 * ITEM_NUM_1 / params.BATCH_SIZE)/10) == 0:
                    if epoch > start:
                        ckpt_name = str(model.model_name + '_' + params.metaName_1 + 
                            '_' + params.metaName_2 + '_' + '_'.join([str(val) for val in paramsList]) + '.ckpt-' + str(epoch))
                        model_saver.save(sess, ckpt_dir + ckpt_name)

        except KeyboardInterrupt:
            if epoch - start > 100:
                ckpt_name = str(model.model_name + '_' + params.metaName_1 + 
                            '_' + params.metaName_2 + '_' + '_'.join([str(val) for val in paramsList]) + '.ckpt-' + str(epoch))
                model_saver.save(sess, ckpt_dir + ckpt_name)


        print("----------saving embedding--------------")
        pickle.dump(u_embeddings_1,userEmbeddingFile_1)
        pickle.dump(i_embeddings_1,itemEmbeddingFile_1)
        userEmbeddingFile_1.close()
        itemEmbeddingFile_1.close()
        pickle.dump(u_embeddings_2,userEmbeddingFile_2)
        pickle.dump(i_embeddings_2,itemEmbeddingFile_2)
        userEmbeddingFile_2.close()
        itemEmbeddingFile_2.close()

def genEigenvector(laplacian_mat, eigenPickleFileName):
    print("----------computing eigen vectors-------------")
    lamda, U = np.linalg.eig(laplacian_mat)
    with open(eigenPickleFileName, 'wb') as pf:
        pickle.dump([lamda,U], pf)
    print("------------eigen vectors settle----------------")
    return lamda, U

def loadEigenvector(laplacian_mat, eigenPickleFileName):
    print("-------------loading the eigen vectors------------")
    with open(eigenPickleFileName, 'rb') as pf:
        eigPara = pickle.load(pf)
        if len(eigPara)>1:
            print("-------Not Empty file! Generating!-------")
            lamda = eigPara[0]
            U = eigPara[1]
        else:
            lamda, U = genEigenvector(laplacian_mat, eigenPickleFileName)
    print("-----------loading eigen vectors successfully!-----------")
    return lamda, U

def adjacient_matrix(graph, self_connection=False):
    num_user = graph.shape[0]
    num_item = graph.shape[1]
    A = np.zeros([num_user+num_item, num_user+num_item], dtype=np.float32)
    A[:num_user, num_user:] = graph
    A[num_user:, :num_user] = graph.T
    if self_connection == True:
        return np.identity(num_user+num_item,dtype=np.float32) + A
    return A

def degree_matrix(adjMat):
    degree = np.sum(adjMat, axis=1, keepdims=False)
    #degree = np.diag(degree)
    return degree


def laplacian_matrix(degree_mat, adjacient_mat, normalized=False):
    if normalized == False:
        return degree_mat - adjacient_mat

    temp = np.dot(np.diag(np.power(degree_mat, -1)), adjacient_mat)
    #temp = np.dot(temp, np.power(self.D, -0.5))
    return np.identity(temp.shape[0],dtype=np.float32) - temp
