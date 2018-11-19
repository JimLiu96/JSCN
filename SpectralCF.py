import tensorflow as tf
import utils as ut
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os.path
# from params import *
# import params



class SpectralCF(object):
    def __init__(self, K, graphLambda, graphU, n_users, n_items, emb_dim, lr, batch_size, decay, DIR):
        self.model_name = 'GraphCF'
        # self.graph = graph
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.K = K
        self.decay = decay

        # self.A = self.adjacient_matrix(self_connection=True)
        # self.D = self.degree_matrix()
        # self.L = self.laplacian_matrix(normalized=True)
        
        # eigPara = []
        
        # #### load or dump the eigen value
        # print("load or dump the eigen value")
        # if os.path.exists(eigenPickleFile):
        #     self.graphLambda, self.U = self.loadEigenvector(eigenPickleFile)
        # else:
        #     self.graphLambda, self.U = self.genEigenvector(eigenPickleFile)    

        # self.graphLambda = np.diag(self.graphLambda)
        # print("the shape of constant")
        # print(self.L.shape, self.graphLambda.shape, self.U.shape)

        # placeholder definition
        # g = tf.Graph()
        # with g.as_default():
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size, ))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.graphLambda = tf.placeholder(tf.float32, shape=(None,
                             None), name='graphLambda')
        self.U = tf.placeholder(tf.float32, shape=(self.n_users+self.n_items, 
                    self.n_users+self.n_items), name="U_mat")
        # self.graphLambda = graphLambda
        # self.U = graphU

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')

        self.filters = []
        for k in range(self.K):
            self.filters.append(
                tf.Variable(
                    tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32)))


        A_hat = tf.matmul(self.U, tf.transpose(self.U)) + tf.matmul(tf.matmul(self.U, self.graphLambda), 
                tf.transpose(self.U))
        # A_hat = tf.matmul(self.U, self.U)+ tf.matmul(tf.matmul(self.U, self.graphLambda), 
                # self.U)

        # A_hat = np.dot(self.U, self.U.T) + np.dot(np.dot(self.U, self.graphLambda), self.U.T)
        # A_hat = A_hat.astype(np.float32)
        #A_hat += np.dot(np.dot(self.U, self.graphLambda_2), self.U.T)
        
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.K):

            embeddings = tf.matmul(A_hat, embeddings)

            #filters = self.filters[k]#tf.squeeze(tf.gather(self.filters, k))
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_embeddings, self.i_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        batch_u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        batch_pos_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        batch_neg_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)

        self.all_ratings = tf.matmul(self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True)

#         print("---------------creating loss-------------")

        self.loss = self.create_bpr_loss(batch_u_embeddings, batch_pos_i_embeddings, batch_neg_i_embeddings)

#         print("---------------loss is set----------------")

        self.opt = tf.train.RMSPropOptimizer(learning_rate=lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings] + self.filters)




    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log_sigmoid(pos_scores - neg_scores)
        # maxi = tf.log_sigmoid(neg_scores)
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        # loss = tf.negative(tf.reduce_sum(maxi)) + self.decay * regularizer
        return loss

    def embeddingSaving(self, loss, u_embeddings, i_embeddings,userOpenFile, itemOpenFile):
        userOpenFile.write(str(loss))
        userOpenFile.write("\n")
#         userNum = self.n_users
        for uidx in range(self.n_users):
            userOpenFile.write(str(u_embeddings[uidx, :]))
        userOpenFile.write("\n")
        itemOpenFile.write(str(loss))
        itemOpenFile.write("\n")
        for iidx in range(self.n_items):
            itemOpenFile.write(str(i_embeddings[iidx, :]))
        itemOpenFile.write("\n")        

    # def genEigenvector(self, eigenPickleFileName):
    #     print("----------computing eigen vectors-------------")
    #     graphLambda, U = np.linalg.eigh(self.L)
    #     with open(eigenPickleFileName, 'wb') as pf:
    #         pickle.dump([graphLambda,U], pf)
    #     print("------------eigen vectors settle----------------")
    #     return graphLambda, U

    # def loadEigenvector(self, eigenPickleFileName):
    #     print("-------------loading the eigen vectors------------")
    #     with open(eigenPickleFileName, 'rb') as pf:
    #         eigPara = pickle.load(pf)
    #         if len(eigPara)>1:
    #             graphLambda = eigPara[0]
    #             U = eigPara[1]
    #         else:
    #             self.genEigenvector(eigenPickleFileName)
    #     print("-----------loading eigen vectors successfully!-----------")
    #     return graphLambda, U

    # def adjacient_matrix(self, self_connection=False):
    #     A = np.zeros([self.n_users+self.n_items, self.n_users+self.n_items], dtype=np.float32)
    #     A[:self.n_users, self.n_users:] = self.graph
    #     A[self.n_users:, :self.n_users] = self.graph.T
    #     if self_connection == True:
    #         return np.identity(self.n_users+self.n_items,dtype=np.float32) + A
    #     return A

    # def degree_matrix(self):
    #     degree = np.sum(self.A, axis=1, keepdims=False)
    #     #degree = np.diag(degree)
    #     return degree


    # def laplacian_matrix(self, normalized=False):
    #     if normalized == False:
    #         return self.D - self.A

    #     temp = np.dot(np.diag(np.power(self.D, -1)), self.A)
    #     #temp = np.dot(temp, np.power(self.D, -0.5))
    #     return np.identity(self.n_users+self.n_items,dtype=np.float32) - temp




        






