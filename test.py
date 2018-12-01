import utils as ut
# from params import *
import params
import load_data
import multiprocessing
import numpy as np
cores = multiprocessing.cpu_count()

print(params.BATCH_SIZE)
# train_user_file_name = 'testTrain.dat'
data_generator_1 = load_data.Data(train_file=params.DIR + params.trainUserFileName_1, test_file=params.DIR+params.testUserFileName_1,batch_size=params.BATCH_SIZE)
data_generator_2 = load_data.Data(train_file=params.DIR + params.trainUserFileName_2, test_file=params.DIR+params.testUserFileName_2,batch_size=params.BATCH_SIZE)
data_generator_all = [data_generator_1, data_generator_2]
# print(params.trainUserFileName_1, params.testUserFileName_1)
# print(params.trainUserFileName_2, params.testUserFileName_2)
USER_NUM_1, ITEM_NUM_1 = data_generator_1.get_num_users_items()
USER_NUM_2, ITEM_NUM_2 = data_generator_2.get_num_users_items()

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    item_num = x[2]

    data_generator = data_generator_all[x[3]]

    training_items = data_generator.train_items[u]
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(item_num))

    test_items = list(all_items - set(training_items))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    recall_20 = ut.recall_at_k(r, 20, len(user_pos_test))
    recall_40 = ut.recall_at_k(r, 40, len(user_pos_test))
    recall_60 = ut.recall_at_k(r, 60, len(user_pos_test))
    recall_80 = ut.recall_at_k(r, 80, len(user_pos_test))
    recall_100 = ut.recall_at_k(r, 100, len(user_pos_test))

    ap_20 = ut.average_precision(r,20)
    ap_40 = ut.average_precision(r, 40)
    ap_60 = ut.average_precision(r, 60)
    ap_80 = ut.average_precision(r, 80)
    ap_100 = ut.average_precision(r, 100)


    return np.array([recall_20,recall_40,recall_60,recall_80,recall_100, ap_20,ap_40,ap_60,ap_80,ap_100])

def testAll(sess, model, users_to_test, feed_dict):
    result_1 = np.array([0.] * 10)
    result_2 = np.array([0.] * 10)
    pool = multiprocessing.Pool(cores)
    batch_size = params.BATCH_SIZE
    #all users needed to test
    test_users_1 = users_to_test[0]
    test_users_2 = users_to_test[1]
    test_user_num_1 = len(test_users_1)
    test_user_num_2 = len(test_users_2)
    item_num_list_1 = [model.n_items_1] * test_user_num_1
    item_num_list_2 = [model.n_items_2] * test_user_num_2
    data_generator_batch_1 = [0] * test_user_num_1
    data_generator_batch_2 = [1] * test_user_num_2
    user_batch_1 = test_users_1
    user_batch_2 = test_users_2
    # user_batch_rating_1, user_batch_rating_2 = sess.run([model.all_ratings_1, model.all_ratings_2], feed_dict=feed_dict)
    user_batch_rating_1, user_batch_rating_2 = sess.run([model.all_ratings_1, model.all_ratings_2], feed_dict=feed_dict)
    # user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
    # user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
    user_batch_rating_uid_1 = zip(user_batch_rating_1, user_batch_1, item_num_list_1, data_generator_batch_1)
    user_batch_rating_uid_2 = zip(user_batch_rating_2, user_batch_2, item_num_list_2, data_generator_batch_2)
    
    batch_result_1 = pool.map(test_one_user, user_batch_rating_uid_1)
    batch_result_2 = pool.map(test_one_user, user_batch_rating_uid_2)

    for re in batch_result_1:
        result_1 += re
    for re in batch_result_2:
        result_2 += re
    pool.close()

    ret_1 = result_1 / test_user_num_1
    ret_2 = result_2 / test_user_num_2
    ret = [list(ret_1), list(ret_2)]
    return ret



# def testAll(sess, model, users_to_test, data_generator, feed_dict):
#     result_1 = np.array([0.] * 10)
#     result_2 = np.array([0.] * 10)
#     data_generator_1 = data_generator[0]
#     data_generator_2 = data_generator[1]
#     pool = multiprocessing.Pool(cores)
#     batch_size = params.BATCH_SIZE
#     #all users needed to test
#     test_users_1 = users_to_test[0]
#     test_users_2 = users_to_test[1]
#     test_user_num_1 = len(test_users_1)
#     test_user_num_2 = len(test_users_2)
#     item_num_list_1 = [model.n_items_1] * params.BATCH_SIZE
#     item_num_list_2 = [model.n_items_2] * params.BATCH_SIZE
#     data_generator_batch_1 = [0] * params.BATCH_SIZE
#     data_generator_batch_2 = [1] * params.BATCH_SIZE
#     index = 0
#     while True:
#         if index >= min(test_user_num_1, test_user_num_2):
#             break
#         user_batch_1 = test_users_1[index:index + batch_size]
#         user_batch_2 = test_users_2[index:index + batch_size]
#         index += batch_size

#         FLAG_1 = False
#         if len(user_batch_1) < batch_size:
#             user_batch_1 += [user_batch_1[-1]] * (batch_size - len(user_batch_1))
#             user_batch_len_1 = len(user_batch_1)
#             FLAG_1 = True
        
#         FLAG_2 = False
#         if len(user_batch_2) < batch_size:
#             user_batch_2 += [user_batch_2[-1]] * (batch_size - len(user_batch_2))
#             user_batch_len_2 = len(user_batch_2)
#             FLAG = True
        
#         user_batch_rating_1, user_batch_rating_2 = sess.run([model.all_ratings_1, model.all_ratings_2], feed_dict=feed_dict)
#         # user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
#         user_batch_rating_uid_1 = zip(user_batch_rating_1, user_batch_1, item_num_list_1, data_generator_batch_1)
#         user_batch_rating_uid_2 = zip(user_batch_rating_2, user_batch_2, item_num_list_2, data_generator_batch_2)
        
#         batch_result_1 = pool.map(test_one_user, user_batch_rating_uid_1)
#         batch_result_2 = pool.map(test_one_user, user_batch_rating_uid_2)

#         if FLAG_1 == True:
#             batch_result_1 = batch_result_1[:user_batch_len_1]
#         if FLAG_2 == True:
#             batch_result_2 = batch_result_2[:user_batch_len_2]
#         for re in batch_result_1:
#             result_1 += re
#         for re in batch_result_2:
#             result_2 += re
#     pool.close()
#     ret_1 = result_1 / test_user_num_1
#     ret_2 = result_2 / test_user_num_2
#     ret = [list(ret_1), list(ret_2)]
#     return ret

# data_generator = load_data.Data(train_file=params.DIR + params.trainUserFileName, test_file=params.DIR+params.testUserFileName,batch_size=params.BATCH_SIZE)
# USER_NUM, ITEM_NUM = data_generator.get_num_users_items()

# def test_one_user(x):
#     # user u's ratings for user u
#     rating = x[0]
#     #uid
#     u = x[1]
#     #user u's items in the training set
#     training_items = data_generator.train_items[u]
#     #user u's items in the test set
#     user_pos_test = data_generator.test_set[u]

#     all_items = set(range(ITEM_NUM))

#     test_items = list(all_items - set(training_items))
#     item_score = []
#     for i in test_items:
#         item_score.append((i, rating[i]))

#     item_score = sorted(item_score, key=lambda x: x[1])
#     item_score.reverse()
#     item_sort = [x[0] for x in item_score]

#     r = []
#     for i in item_sort:
#         if i in user_pos_test:
#             r.append(1)
#         else:
#             r.append(0)

#     recall_20 = ut.recall_at_k(r, 20, len(user_pos_test))
#     recall_40 = ut.recall_at_k(r, 40, len(user_pos_test))
#     recall_60 = ut.recall_at_k(r, 60, len(user_pos_test))
#     recall_80 = ut.recall_at_k(r, 80, len(user_pos_test))
#     recall_100 = ut.recall_at_k(r, 100, len(user_pos_test))

#     ap_20 = ut.average_precision(r,20)
#     ap_40 = ut.average_precision(r, 40)
#     ap_60 = ut.average_precision(r, 60)
#     ap_80 = ut.average_precision(r, 80)
#     ap_100 = ut.average_precision(r, 100)


#     return np.array([recall_20,recall_40,recall_60,recall_80,recall_100, ap_20,ap_40,ap_60,ap_80,ap_100])

# def testAll(sess, model, users_to_test):
#     result = np.array([0.] * 10)
#     pool = multiprocessing.Pool(cores)
#     batch_size = params.BATCH_SIZE
#     #all users needed to test
#     test_users = users_to_test
#     test_user_num = len(test_users)
#     index = 0
#     while True:
#         if index >= test_user_num:
#             break
#         user_batch = test_users[index:index + batch_size]
#         index += batch_size
#         FLAG = False
#         if len(user_batch) < batch_size:
#             user_batch += [user_batch[-1]] * (batch_size - len(user_batch))
#             user_batch_len = len(user_batch)
#             FLAG = True
#         user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch,})
#         user_batch_rating_uid = zip(user_batch_rating, user_batch)
#         batch_result = pool.map(test_one_user, user_batch_rating_uid)

#         if FLAG == True:
#             batch_result = batch_result[:user_batch_len]
#         for re in batch_result:
#             result += re

#     pool.close()
#     ret = result / test_user_num
#     ret = list(ret)
#     return ret