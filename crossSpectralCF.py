import tensorflow as tf
import utils as ut
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os.path



class SpectralCF(object):
    def __init__(self, K, M, n_users_1, n_users_2, n_items_1, n_items_2, commonUser, emb_dim, lr, batch_size, decay, DIR):
        self.model_name = 'crossGraphCF'
        # self.graph = graph
        self.n_users_1 = n_users_1
        self.n_items_1 = n_items_1
        self.n_users_2 = n_users_2
        self.n_items_2 = n_items_2
        self.common_user_pair = commonUser
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.K = K
        self.decay = decay
        self.bpr_loss = [0]*M
        var_list = []

        self.users_1 = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items_1 = tf.placeholder(tf.int32, shape=(self.batch_size, ))
        self.neg_items_1 = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.graphLambda_1 = tf.placeholder(tf.float32, shape=(None,
                             None), name='graphLambda_1')
        self.U_1 = tf.placeholder(tf.float32, shape=(self.n_users_1+self.n_items_1, 
                    self.n_users_1 + self.n_items_1), name="U_mat_1")

        self.user_embeddings_1 = tf.Variable(
            tf.random_normal([self.n_users_1, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings_1 = tf.Variable(
            tf.random_normal([self.n_items_1, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')
        var_list.append(self.user_embeddings_1)
        var_list.append(self.item_embeddings_1)

        self.filters_1 = []
        for k in range(self.K):
            self.filters_1.append(
                tf.Variable(
                    tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32)))

        A_hat_1 = tf.matmul(self.U_1, tf.transpose(self.U_1)) + tf.matmul(tf.matmul(self.U_1, self.graphLambda_1), 
                tf.transpose(self.U_1))

        var_list.append(self.filters_1)

        embeddings_1 = tf.concat([self.user_embeddings_1, self.item_embeddings_1], axis=0)
        all_embeddings_1 = [embeddings_1]
        for k in range(0, self.K):
            embeddings_1 = tf.matmul(A_hat_1, embeddings_1)
            embeddings_1 = tf.nn.sigmoid(tf.matmul(embeddings_1, self.filters_1[k]))
            all_embeddings_1 += [embeddings_1]
        all_embeddings_1 = tf.concat(all_embeddings_1, 1)
        self.u_embeddings_1, self.i_embeddings_1 = tf.split(all_embeddings_1, [self.n_users_1, self.n_items_1], 0)

        # self.u_embeddings_1 = tf.nn.embedding_lookup(self.u_embeddings_1, self.users_1)
        # self.pos_i_embeddings_1 = tf.nn.embedding_lookup(self.i_embeddings_1, self.pos_items_1)
        # self.neg_i_embeddings_1 = tf.nn.embedding_lookup(self.i_embeddings_1, self.neg_items_1)
        batch_u_embeddings_1 = tf.nn.embedding_lookup(self.u_embeddings_1, self.users_1)
        batch_pos_i_embeddings_1 = tf.nn.embedding_lookup(self.i_embeddings_1, self.pos_items_1)
        batch_neg_i_embeddings_1 = tf.nn.embedding_lookup(self.i_embeddings_1, self.neg_items_1)

        self.all_ratings_1 = tf.matmul(self.u_embeddings_1, self.i_embeddings_1, transpose_a=False, transpose_b=True)

        # self.bpr_loss[0] = self.create_bpr_loss(self.u_embeddings_1, self.pos_i_embeddings_1, self.neg_i_embeddings_1)
        self.bpr_loss[0] = self.create_bpr_loss(batch_u_embeddings_1, batch_pos_i_embeddings_1, batch_neg_i_embeddings_1)

        # second graph
        if M > 1:
            self.users_2 = tf.placeholder(tf.int32, shape=(self.batch_size,))
            self.pos_items_2 = tf.placeholder(tf.int32, shape=(self.batch_size, ))
            self.neg_items_2 = tf.placeholder(tf.int32, shape=(self.batch_size,))
            self.graphLambda_2 = tf.placeholder(tf.float32, shape=(None,
                                 None), name='graphLambda_2')
            self.U_2 = tf.placeholder(tf.float32, shape=(self.n_users_2+self.n_items_2, 
                        self.n_users_2 + self.n_items_2), name="U_mat_2")

            self.user_embeddings_2 = tf.Variable(
                tf.random_normal([self.n_users_2, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings_2')
            self.item_embeddings_2 = tf.Variable(
                tf.random_normal([self.n_items_2, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings_2')

            var_list.append(self.user_embeddings_2)
            var_list.append(self.item_embeddings_2)


            self.filters_2 = []
            for k in range(self.K):
                self.filters_2.append(
                    tf.Variable(
                        tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32)))
            var_list.append(self.filters_2)
            A_hat_2 = tf.matmul(self.U_2, tf.transpose(self.U_2)) + tf.matmul(tf.matmul(self.U_2, self.graphLambda_2), 
                    tf.transpose(self.U_2))
            
            embeddings_2 = tf.concat([self.user_embeddings_2, self.item_embeddings_2], axis=0)
            all_embeddings_2 = [embeddings_2]
            for k in range(0, self.K):
                embeddings_2 = tf.matmul(A_hat_2, embeddings_2)
                embeddings_2 = tf.nn.sigmoid(tf.matmul(embeddings_2, self.filters_2[k]))
                all_embeddings_2 += [embeddings_2]

            all_embeddings_2 = tf.concat(all_embeddings_2, 1)
            self.u_embeddings_2, self.i_embeddings_2 = tf.split(all_embeddings_2, [self.n_users_2, self.n_items_2], 0)

            # self.u_embeddings_2 = tf.nn.embedding_lookup(self.u_embeddings_2, self.users_2)
            # self.pos_i_embeddings_2 = tf.nn.embedding_lookup(self.i_embeddings_2, self.pos_items_2)
            # self.neg_i_embeddings_2 = tf.nn.embedding_lookup(self.i_embeddings_2, self.neg_items_2)
            
            self.u_embeddings_2 = tf.nn.embedding_lookup(self.u_embeddings_2, self.users_2)
            self.pos_i_embeddings_2 = tf.nn.embedding_lookup(self.i_embeddings_2, self.pos_items_2)
            self.neg_i_embeddings_2 = tf.nn.embedding_lookup(self.i_embeddings_2, self.neg_items_2)

            self.all_ratings_2 = tf.matmul(self.u_embeddings_2, self.i_embeddings_2, transpose_a=False, transpose_b=True)

            self.bpr_loss[1] = self.create_bpr_loss(self.u_embeddings_2, self.pos_i_embeddings_2, self.neg_i_embeddings_2)
    #         print("---------------loss is set----------------")


        # Loss
        self.bpr_loss_all = sum(self.bpr_loss)
        self.loss = self.bpr_loss_all + 0

        self.opt = tf.train.RMSPropOptimizer(learning_rate=lr)

        self.updates = self.opt.minimize(self.loss, var_list=var_list)
        # self.updates = self.opt.minimize(self.bpr_loss_all, var_list=var_list)



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

    def create_joint_loss(self):
        return


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




        






