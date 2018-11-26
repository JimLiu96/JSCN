import json
import gzip
import pickle

def split(item_list):
    train = item_list[:int(len(item_list) * 0.8)]
    test = item_list[int(len(item_list) * 0.8):]

    return train, test

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def processing(ratingFileName, num_items = 5, train_items = 5, num_users = 5, rating_score=5):
    # num_items is the threshold for the minimum items for one user
    # num_users is the threshold for the minimum users buying one item
    u2items = {}
    item2id = {}
    id2user = {}
    id2item = {}
    users, items = set(), set()
    ratings = 0
    #for l in parse("reviews_Video_Games_5.json.gz"):
    #    data = json.loads(l)
    #    u = data['reviewerID']
    #    i = data['asin']
    #    r = data['overall']
    f = open(ratingFileName)
    for l in f.readlines():
        u,i,r = l.split(',')[0:3]
        r = float(r)
        users.add(u)
        items.add(i)
        ratings += 1
        if r >= rating_score:
            if u in u2items:
                u2items[u].append(i)
            else:
                u2items[u] = [i]

    print('total Statistic:',len(users),len(items),ratings)
    u2items = {k:v for k,v in u2items.items() if len(v) >= num_items}
    i2users = filterItems(u2items, num_users)
    iid = 0
    for k,items in u2items.items():
        temp = []
        for i in items:
            if i in i2users:
                if i not in item2id:
                    item2id[i] = iid
                    iid += 1
                temp.append(item2id[i])
        u2items[k] = temp
    u2items = {k:v for k,v in u2items.items() if len(v) >= num_items}
    id2item = {idNum:itemNum for itemNum,idNum in item2id.items()}
    num_ratings = 0
    for u in u2items:
        num_ratings += len(u2items[u])
    uid = 0
    metaName = ratingFileName.split('.')[0]
    trainFileName = metaName + '_train_user.dat'
    testFileName =  metaName + '_test_user.dat'
    userDumpFileName = metaName +'_id2user.pickle'
    itemDumpFileName = metaName + '_id2item.pickle'

    with open(trainFileName, 'w') as f_train:
        with open(testFileName, 'w') as f_test:
            for k, v in u2items.items():
                if k not in id2user:
                    id2user[uid] = k
                if len(v) <= train_items:
                    # print("less than %d items put into train only" %( train_items))
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in v]) + '\n')
                else:
                    train, test = split(v)
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in train]) + '\n')
                    f_test.write(str(uid) + ' ' + ' '.join([str(i) for i in test]) + '\n')
                uid += 1
    print('starting dump the id of users and items')
    with open(userDumpFileName,'wb') as userDumpFile:
        pickle.dump(id2user,userDumpFile)
    with open(itemDumpFileName,'wb') as itemDumpFile:
        pickle.dump(id2item,itemDumpFile)

    print('Remaning:',uid,iid)
    print(num_ratings/(uid*iid))

def filterItems(user2items, num_users):
    i2userTemp = {}
    for user in user2items:
        itemList = user2items[user]
        for item in itemList:
            if item not in i2userTemp:
                i2userTemp[item] = [user]
            else:
                i2userTemp[item].append(user)
    i2users = {i:u for i,u in i2userTemp.items() if len(u)>=num_users}
    return i2users

def loadID2UandI(metaName):
    id2userFileName = metaName + '_id2user.pickle'
    id2itemFileName = metaName + '_id2item.pickle'
    with open(id2userFileName ,'rb') as id2userFile:
        id2user = pickle.load(id2userFile)
    with open(id2itemFileName, 'rb') as id2itemFile:
        id2item = pickle.load(id2itemFile)
    return id2user, id2item

def commonUserItem(metaName1, metaName2):
    id2userMetaName1, id2itemMetaName1 = loadID2UandI(metaName1)
    id2userMetaName2, id2itemMetaName2 = loadID2UandI(metaName2)
    userInMetaName1 = [u for uid,u in id2userMetaName1.items() ]
    userInMetaName2 = [u for uid,u in id2userMetaName2.items() ]
    itemInMetaName1 = [i for iid,i in id2itemMetaName1.items()]
    itemInMetaName2 = [i for iid,i in id2itemMetaName2.items()]
    sameUser = list(set(userInMetaName1).intersection(userInMetaName2))
    sameItem = list(set(itemInMetaName1).intersection(itemInMetaName2))
    return sameUser, sameItem

def commonUserIDpair(metaName1, metaName2):
    # return [(IDin1, IDin2)]
    commonUserIdxPair = []
    id2userMetaName1, _ = loadID2UandI(metaName1)
    id2userMetaName2, _ = loadID2UandI(metaName2)
    user2idMetaName1 = {v:k for k,v in id2userMetaName1.items()}
    user2idMetaName2 = {v:k for k,v in id2userMetaName2.items()}
    sameUser,_ = commonUserItem(metaName1,metaName2)
    for userName in sameUser:
        userID_1 = user2idMetaName1[userName]
        userID_2 = user2idMetaName2[userName]
        commonUserIdxPair.append((userID_1,userID_2))
    commonUserIDpairDumpFileName = metaName1 + '-' + metaName2.split("/")[-1] + '-' + 'commonUserIDpair.pickle'
    with open(commonUserIDpairDumpFileName, 'wb') as dumpF:
        pickle.dump(commonUserIdxPair, dumpF)
    return commonUserIdxPair
