from back_SpectralCF import *
from test import *
import tensorflow as tf
import pickle
from params import * 
import load_data
# from SpectralCF import * 



def modelTrain(testState = False):
    # userEmbeddingDIR = './embedding/userEmbedding.txt'
    userEmbeddingFile = open(userEmbeddingDIR ,'wb')
    # itemEmbeddingDIR = './embedding/itemEmbedding.txt'
    itemEmbeddingFile = open(itemEmbeddingDIR ,'wb')
#     graph=data_generator.R
    model = SpectralCF(K=K, graph=data_generator.R, n_users=USER_NUM, n_items=ITEM_NUM, emb_dim=EMB_DIM,
                     lr=LR, decay=DECAY, batch_size=BATCH_SIZE,DIR=DIR)
    print(model.model_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("session created")
    sess.run(tf.global_variables_initializer())
    print("-------------------initialization finished-------------------")

    for epoch in range(N_EPOCH):
#         print("-----------Running Epoch %d-----------" % epoch)
        users, pos_items, neg_items = data_generator.sample()
#         print(len(users), len(pos_items), len(neg_items))
        _, loss, u_embeddings, i_embeddings = sess.run([model.updates, model.loss, 
          model.user_embeddings, model.item_embeddings],
          feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
#         print(u_embeddings)
#         userEmbeddingNumpy = model.u_embeddings.eval(session=sess)
#         print(u_embeddings.shape)
#         print(i_embeddings.shape)
        if epoch%1 == 0:
            print('Epoch %d training loss %f' % (epoch, loss))
            if testState == True:
                users_to_test = list(data_generator.test_set.keys())
                ret = testAll(sess, model, users_to_test)
                print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
                        % (ret[0],ret[1],ret[2],ret[3],ret[4]))
                print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
                    % (ret[5], ret[6], ret[7], ret[8], ret[9]))
    print("----------saving embedding--------------")
    pickle.dump(u_embeddings,userEmbeddingFile)
    pickle.dump(i_embeddings,itemEmbeddingFile)
    userEmbeddingFile.close()
    itemEmbeddingFile.close()

