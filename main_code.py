import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import warnings
from surprise import Dataset
from surprise import Reader
import math as m
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

N = 1682 # no. of items
M = 943 # no. of users
num_loops = 2 # no. of times to run the algorithm
dataMat = [[0 for i in range(M)] for j in range(N)] #store ratings for user-item

def getInt(x, val): #convert float predictions to int
    global min_thresArr
    arr=min_thresArr
    if type(x) == float: 
        if val == 0:
            i=0
            while i<4 and x>arr[i]:
                i+=1
            return i+1
        else :                  
            x = round(x,2)
            if x-m.floor(x) >= 0.5:
                return int(x)+1
            return int(x)
    return x
    
def getRoundV(x, arr): #conert float predictions to int based on threshold array
    i=0 
    if type(x) == float:
        while i<4 and x>arr[i]:
            i+=1
        return i+1
    return x

def createDf(x, y, l): #generate dataframe
    df={'item':[],'user':[], 'rating':[]}
    for i in range(l):
        df['user'].append(x[i][0])
        df['item'].append(x[i][1])
        df['rating'].append(y[i])
    return df

def genFile(mat,path, r, c): #generate output file
    print("\n\nGenerating output file {}....".format(path))
    with open(path, "w+") as f:
        for i in range(c):
            for j in range(r):
                f.write(" ".join([str(i+1), str(j+1), str(mat[j][i])]))
                if i == c-1 and j == r-1: 
                    break
                else:
                    f.write("\n")
                    
    print("\nOutput file generated !!")

def fillMatrix(model, isThres): #fill remaining rating values for user-item combination
    global dataMat,N,M, cold_itm, avg_rat
    par = 0 if isThres is True else 1
    print("\n\nFilling matrix....")
    for i in range(N):
        for j in range(M):
            if dataMat[i][j]==0:
                if i in cold_itm:
                    dataMat[i][j] = avg_rat[j]
                else:
                    dataMat[i][j] = getInt(model.predict(j+1,i+1).est,par)
    print("\nMatrix filled !!")

def getAcc(pred,act): #Calculate accuracy
    pos,neg=[0,0]
    for i in range(len(pred)):
        if pred[i] == act[i]:
            pos+=1
        else:
            neg+=1
    return pos/(pos+neg)
 
def getRMSE(pred,act): #calculate RMSE value
    global testlen
    res=sum([(pred[i]-act[i])**2 for i in range(testlen)])/(testlen)
    return res**(0.5)

def getMAE(pred,act): #calculate MAE value
    global testlen
    res=sum([abs(pred[i]-act[i]) for i in range(testlen)])/(testlen)
    return res
                   
def KNNPred(data): #KNN Means algorithm
    print("\nTraining KNN Means model..\n")
    global x_test, y_test, testlen, trainlen, y_train, model_params, X, Y,  avg_rat, cold_itm
    options = model_params[0]
    knnModel = KNNWithMeans(sim_options=options) 
    knnModel_1 = KNNWithMeans()
    train = data.build_full_trainset()    
    knnModel.fit(train)
    print("\nTraining done..\nPrediction started..")
    knnModel_1.fit(train)
    #y_pred_w_m = [knnModel.predict(x_test[i][0], x_test[i][1]).est for i in range(testlen)]
    #y_pred_wo_m = [knnModel_1.predict(x_test[i][0], x_test[i][1]).est for i in range(testlen)]
    y_pred_w_m = [0 for i in range(testlen)]
    y_pred_wo_m = [0 for i in range(testlen)]
    kk=0
    for i in x_test:
        if i[1]-1 in cold_itm:
            y_pred_w_m[kk] = avg_rat[i[0]-1]
            y_pred_wo_m[kk] = avg_rat[i[0]-1]
        else:
            y_pred_w_m[kk] = knnModel.predict(i[0], i[1]).est
            y_pred_wo_m[kk] = knnModel_1.predict(i[0], i[1]).est
        kk+=1
    #y_pred_train = [knnModel_1.predict(x_train[i][0], x_train[i][1]).est for i in range(trainlen)]
    #y_pred_tot = [knnModel_1.predict(X[i][0], X[i][1]).est for i in range(trainlen+testlen)]
    print("\nPrediction done..\n")
    return [y_pred_w_m, y_pred_wo_m, knnModel, knnModel_1] #, y_pred_train, y_pred_tot

def svdPP(data): #SVDPP algorithm
    print("\nTraining SVDPP model..\n")
    global x_test, y_test, testlen, trainlen, model_params, x_train, y_train, X, Y, avg_rat, cold_itm
    p1,p2,p3=[model_params[1]['n_epochs'],model_params[1]['lr_all'],model_params[1]['reg_all']]
    svdModel = SVDpp(n_epochs=p1,lr_all=p2,reg_all=p3)
    svdModel.fit(data.build_full_trainset())
    print("\nTraining done..\nPrediction started..")
    test=[(x_test[i][0],x_test[i][1],y_test[i]) for i in range(testlen)]
    #train_=[(x_train[i][0],x_train[i][1],y_train[i]) for i in range(trainlen)]
    #total_=[(X[i][0],X[i][1],Y[i]) for i in range(trainlen+testlen)]
    predict = svdModel.test(test)
    
    
    #trainset, testset = t_t_s(data, test_size=.25)
    svdModel_1 = SVDpp()
    svdModel_1.fit(data.build_full_trainset())
    predict1 = svdModel_1.test(test)
    #predict_train = svdModel_1.test(train_)
    #predict_tot = svdModel_1.test(total_)
    usrA=[int(i[0])-1 for i in predict]
    itmA=[int(i[1])-1 for i in predict]
    res=[i[3] for i in predict]
    res1=[i[3] for i in predict1]
    for i in range(testlen):
        if itmA[i] in cold_itm:
            res[i] = avg_rat[usrA[i]]
            res1[i] = avg_rat[usrA[i]]
    #restrain=[i[3] for i in predict_train]
    print("\nPrediction done..\n")
    return [res, res1, svdModel, svdModel_1] #,restrain, predict_tot

def compAlgos(data): #Compare MAE, RMSE values for different algorithms
    print("\nLet us compare performance of KNN and SVD algorithms\n")
    #KNN Algos
    knn_Basic = cross_validate(KNNBasic(), data, cv=5, n_jobs=5, verbose=False)
    knn_means = cross_validate(KNNWithMeans(), data, cv=5, n_jobs=5, verbose=False)
    knn_z = cross_validate(KNNWithZScore(), data, cv=5, n_jobs=5, verbose=False)

    #SVD Algos
    svd = cross_validate(SVD(), data, cv=5, n_jobs=5, verbose=False)
    svdpp = cross_validate(SVDpp(), data, cv=5, n_jobs=5, verbose=False)
    
    print('\nKNN Basic: RMSE: {}, MAE: {}'.format(knn_Basic['test_rmse'].mean(), knn_Basic['test_mae'].mean()))
    print('\nKNN Means: RMSE: {}, MAE: {}'.format(knn_means['test_rmse'].mean(), knn_means['test_mae'].mean()))
    print('\nKNN Z Score: RMSE: {}, MAE: {}'.format(knn_z['test_rmse'].mean(), knn_z['test_mae'].mean()))
    
    print('\nSVD: RMSE: {}, MAE: {}'.format(svd['test_rmse'].mean(), svd['test_mae'].mean()))
    print('\nSVD ++: RMSE: {}, MAE: {}'.format(svdpp['test_rmse'].mean(), svdpp['test_mae'].mean()))

    print('\nBoth SVDs perform better on the dataset\n')
    print('\nWe will test with KNN means from KNN family and SVDPP from svd family\n')

def gridS(data): #Find best parameters for KNN and SVDPP
    print('\nLet us check best parameters for KNN Means algorithm\n')
    options = {
    "name": ["msd", "cosine"],
    "min_support": [2, 3, 4, 5],
    "user_based": [False, True],
    }
    
    knn_grid = {"sim_options": options}
    knn = GridSearchCV(KNNWithMeans, knn_grid, measures=["rmse", "mae", "mse"], cv=5,n_jobs=5)
    knn.fit(data)
    print("\nKNN Means Analysis\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(knn.best_score["rmse"],knn.best_score["mae"],knn.best_score["mse"]))
    print("\nBest Combination of Parameters\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(knn.best_params["rmse"],knn.best_params["mae"],knn.best_params["mse"]))
    
    
    print('\nWe will see which options are best fit for SVDPP algorithm')
    svd_grid = {
    "n_epochs": [5, 10, 15, 20, 25],
    "lr_all": [0.002, 0.005, 0.008, 0.009],
    "reg_all": [0.4, 0.6, 0.8]
    }
    
    '''svd = GridSearchCV(SVD, svd_grid, measures=["rmse", "mae", "mse"], cv=5, n_jobs=5)
    svd.fit(data) 
    print("\nSVD Analysis\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(svd.best_score["rmse"],svd.best_score["mae"],svd.best_score["mse"]))
    print("\nBest Combination of Parameters\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(svd.best_params["rmse"],svd.best_params["mae"],svd.best_params["mse"]))'''
    
    svdpp = GridSearchCV(SVDpp, svd_grid, measures=["rmse", "mae", "mse"], cv=5, n_jobs=5)
    svdpp.fit(data) 
    print("\nSVDpp Analysis\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(svdpp.best_score["rmse"],svdpp.best_score["mae"],svdpp.best_score["mse"]))
    print("\nBest Combination of Parameters\n")
    print("\nRMSE: {}, MAE: {}, MSE: {}\n".format(svdpp.best_params["rmse"],svdpp.best_params["mae"],svdpp.best_params["mse"]))

    print("\nWe will train model based on parameter values best suited for RMSE reduction\n")
    return [knn.best_params["rmse"]["sim_options"],svdpp.best_params["rmse"]]

def getThresArr(): # return arrays for calculating threshold values

    a2=[round(i,1) for i in list(np.arange(1.1,2,0.1))]
    a3=[round(i,1) for i in list(np.arange(2.1,3,0.1))]
    a4=[round(i,1) for i in list(np.arange(3.1,4,0.1))]
    a5=[round(i,1) for i in list(np.arange(4.1,5,0.1))]   
    return [a2,a3,a4,a5]

X=[] #stores user and item
Y=[] #stores ratings
cnt = 0
#fill X and Y from input file
with open('train.txt') as f:
    for line in f:
        line = line.strip()
        arr=[int(i) for i in line.split(" ")]
        X.append([arr[0],arr[1]])
        Y.append(arr[2])
        cnt+=1
usr=list(set([X[i][0] for i in range(len(X))]))
itm = list(set([X[i][1] for i in range(len(X))]))
usrlen = len(usr)
itmlen = len(itm)
#print("user len:{}, item len:{}, cond:{}".format(usrlen,itmlen, itmlen != N and usrlen != M))

cntArr=[0,0,0,0,0]
min_RMSE_model=[] #stores best model with least RMSE value
min_thresArr=[] #stores best threshold values
minLoop=0 # loop in which best fit model is found

params_knn={'name': 'msd', 'min_support': 2, 'user_based': False} #calculated using GridSearchCV in function gridS
params_svd={'n_epochs':25, 'lr_all': 0.009, 'reg_all': 0.4} #calculated using GridSearchCV in function gridS
model_params=[params_knn, params_svd]
isThresUsed=True
bestFitModel = 'KNN Means'
items=[[0 for i in range(5)]for j in range(N)]
users=[[0 for i in range(5)]for j in range(M)]

# Fill data matrix
for i in range(cnt):
    dataMat[X[i][1]-1][X[i][0]-1] = Y[i]
    cntArr[Y[i]-1]+=1
    items[X[i][1]-1][Y[i]-1]+=1
    users[X[i][0]-1][Y[i]-1]+=1

cold_itm = [i for i in range(N) if sum(items[i]) == 0]
avg_rat=[0 for i in range(M)]
k=0
for i in users:
    h = [i[j]*(j+1) for j in range(5)]
    avg_rat[k] = sum(h)/sum(i)
    if round(avg_rat[k],2) - int(avg_rat[k]) >=0.5:
        avg_rat[k] = int(avg_rat[k]) + 1
    else:
        avg_rat[k] = int(avg_rat[k])
    k+=1

origF = createDf(X,Y,len(X)) #create dataframe
    
#Run algorithms for num_loops times and get the best fit model with minimum RMSE value
for test_loop_cnt in range(1,num_loops + 1): 
    x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.25, random_state=42)
    trainlen = len(y_train)
    testlen = len(y_test)
    usr=list(set([x_train[i][0] for i in range(trainlen)]))
    itm = list(set([x_train[i][1] for i in range(trainlen)]))
    usrlen = len(usr)
    itmlen = len(itm)
    #print("user len:{}, item len:{}, cond:{}".format(usrlen,itmlen, itmlen != N and usrlen != M))
    '''while not (itmlen != u_i and usrlen != u_u):
    #train-test split with 0.25 test size
        x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.25, random_state=42)
        trainlen = len(y_train)
        testlen = len(y_test)
        usr=list(set([x_train[i][0] for i in range(trainlen)]))
        itm = list(set([x_train[i][1] for i in range(trainlen)]))
        usrlen = len(usr)
        itmlen = len(itm)
        print("user len:{}, item len:{}".format(len(usr),len(itm)))'''
    #create training dataframe
    dataF= createDf(x_train,y_train,trainlen)
    d_t = [dataF,origF]
    df = [pd.DataFrame(i) for i in d_t] 
    reader = Reader(rating_scale=(1, 5))
    
    # Loads Pandas dataframe
    data = [Dataset.load_from_df(df[i][["user", "item", "rating"]], reader) for i in range(2)]
    
    # Following 3 lines are to determine which model would be suitable and find corrosponding best parameters
    #The parameter values are already stored in model_params at the start
    #The excecution time of this code is approx. 1 hour, 
    #hence I seperately run this code and stored the param values 
    '''if test_loop_cnt == 1:
        compAlgos(data[1])
        model_params = gridS(data[1]) #comment this in case of speeding execution '''
    
    print("Loop {}\n".format(test_loop_cnt))
    
    knModelWM, knModelWoM, knn_w_p, knn_wo_p = KNNPred(data[0]) #, knn_train, knn_tot
    svdModelWM, svdModelWoM, svd_w_p, svd_wo_p = svdPP(data[0]) #, svd_train, svd_tot
    
    thresArr=getThresArr()
    knn_thres={'mae':{'with_param':[], 'without_param':[]}, 'rmse':{'with_param':[], 'without_param':[]}}
    svd_thres={'mae':{'with_param':[], 'without_param':[]}, 'rmse':{'with_param':[], 'without_param':[]}}
    knn_Arr={'mae':{'with_param':[], 'without_param':[]}, 'rmse':{'with_param':[], 'without_param':[]}}
    svd_Arr={'mae':{'with_param':[], 'without_param':[]}, 'rmse':{'with_param':[], 'without_param':[]}}
    accuracy_score={'knn':{'with_param':[], 'without_param':[]},'svd':{'with_param':[], 'without_param':[]}}
    print("\nCalculating best threshold values..\n")
    loop_c=0
    for model in [knModelWM, knModelWoM, svdModelWM, svdModelWoM]:
        resArr=[[],[]]
        minrmse,minmae=-1,-1
        
        for i in thresArr[0]:
            for j in thresArr[1]:
                for k in thresArr[2]:
                    for l in thresArr[3]:
                        t=[i,j,k,l]
                        resA=[getRoundV(x,t) for x in model]
                        rmsT=getRMSE(resA,y_test)
                        maT=getMAE(resA,y_test)
                        if minrmse == -1 or rmsT < minrmse:
                            minrmse=rmsT
                            resArr[0]=t
                    
                        if minmae == -1 or maT < minmae:
                            minmae = maT
                            resArr[1]=t
        
        minmae = round(minmae,4)
        minrmse = round(minrmse,4)
        isEmp=False
        modName=knn_w_p
        bfm = 'KNN Means'
        if len(min_RMSE_model) == 0:
            isEmp = True
        if loop_c == 0:
            knn_Arr['mae']['with_param'].append(minmae)
            knn_Arr['rmse']['with_param'].append(minrmse)
            knn_thres['mae']['with_param'] = resArr[1]
            knn_thres['rmse']['with_param'] = resArr[0]
            accuracy_score['knn']['with_param'].append((round(getAcc([getRoundV(rrr,resArr[0]) for rrr in model],y_test),4)))
            modName = knn_w_p
            bfm = 'KNN Means'
        elif loop_c == 1:
            knn_Arr['mae']['without_param'].append(minmae)
            knn_Arr['rmse']['without_param'].append(minrmse)
            knn_thres['mae']['without_param'] = resArr[1]
            knn_thres['rmse']['without_param'] = resArr[0]
            accuracy_score['knn']['without_param'].append((round(getAcc([getRoundV(rrr,resArr[0]) for rrr in model],y_test),4)))
            modName=knn_wo_p
            bfm = 'KNN Means'
        elif loop_c == 2:
            svd_Arr['mae']['with_param'].append(minmae)
            svd_Arr['rmse']['with_param'].append(minrmse)
            svd_thres['mae']['with_param'] = resArr[1]
            svd_thres['rmse']['with_param'] = resArr[0]
            accuracy_score['svd']['with_param'].append((round(getAcc([getRoundV(rrr,resArr[0]) for rrr in model],y_test),4)))
            modName = svd_w_p
            bfm = 'SVDPP'
        else:
            svd_Arr['mae']['without_param'].append(minmae)
            svd_Arr['rmse']['without_param'].append(minrmse)
            svd_thres['mae']['without_param'] = resArr[1]
            svd_thres['rmse']['without_param'] = resArr[0]
            accuracy_score['svd']['without_param'].append((round(getAcc([getRoundV(rrr,resArr[0]) for rrr in model],y_test),4)))
            modName = svd_wo_p
            bfm = 'SVDPP'
        if isEmp:
            min_RMSE_model=[minrmse,modName]
            min_thresArr=resArr[0]
            minLoop=test_loop_cnt
            isThresUsed=True
            bestFitModel=bfm
        else:
            if minrmse < min_RMSE_model[0]:
                min_RMSE_model=[minrmse,modName]
                min_thresArr=resArr[0]
                isThresUsed=True
                minLoop=test_loop_cnt
                bestFitModel=bfm
        loop_c+=1
    
    
    knRMSE=[getRMSE(i,y_test) for i in [knModelWM, knModelWoM]]
    knMAE=[getMAE(i,y_test) for i in [knModelWM, knModelWoM]]
    
    svdRMSE=[getRMSE(i,y_test) for i in [svdModelWM, svdModelWoM]]
    svdMAE=[getMAE(i,y_test) for i in [svdModelWM, svdModelWoM]]
    
    #knRMSE_train=[getInt(i,0) for i in [knn_train]]
    #svdRMSE_train=[getInt(i,y_train) for i in [svd_train]]
    
    
    
    k=1 
    #Generate integer values without using threshold values, by simple converting float values to closest integer
    knMWM_int, knMWoM_int=[getInt(i,k) for i in knModelWM],[getInt(i,k) for i in knModelWoM]
    svdMWM_int, svdMWoM_int=[getInt(i,k) for i in svdModelWM],[getInt(i,k) for i in svdModelWoM]
    
    accr_int=[round(getAcc(rrr,y_test),4) for rrr in [knMWM_int,knMWoM_int,svdMWM_int,svdMWoM_int]]
    
    knRMSE_int=[round(getRMSE(i,y_test),4) for i in [knMWM_int, knMWoM_int]]
    knMAE_int=[round(getMAE(i,y_test),4) for i in [knMWM_int, knMWoM_int]]
    
    svdRMSE_int=[round(getRMSE(i,y_test),4) for i in [svdMWM_int, svdMWoM_int]]
    svdMAE_int=[round(getMAE(i,y_test),4) for i in [svdMWM_int, svdMWoM_int]]
    
    knn_Arr['mae']['with_param'].append(knRMSE_int[0])
    knn_Arr['mae']['without_param'].append(knRMSE_int[1])
    knn_Arr['rmse']['with_param'].append(knRMSE_int[0])
    knn_Arr['rmse']['without_param'].append(knRMSE_int[1])
    accuracy_score['knn']['with_param'].append(accr_int[0])    
    accuracy_score['knn']['without_param'].append(accr_int[1])
    
    svd_Arr['mae']['with_param'].append(svdMAE_int[0])
    svd_Arr['mae']['without_param'].append(svdMAE_int[1])
    svd_Arr['rmse']['with_param'].append(svdRMSE_int[0])
    svd_Arr['rmse']['without_param'].append(svdRMSE_int[1])
    accuracy_score['svd']['with_param'].append(accr_int[2])    
    accuracy_score['svd']['without_param'].append(accr_int[3])
    
    #isThresUsed = True
    for i in range(2):
        if knRMSE_int[i] < min_RMSE_model[0]:
            if i == 0:
                min_RMSE_model=[knRMSE_int[i], knn_w_p]
                minLoop=test_loop_cnt
            else:
                min_RMSE_model=[knRMSE_int[i], knn_wo_p]
                minLoop=test_loop_cnt
            isThresUsed=False
            bestFitModel='KNN Means'
            
    for i in range(2):
        if svdRMSE_int[i] < min_RMSE_model[0]:
            if i == 0:
                min_RMSE_model=[svdRMSE_int[i], svd_w_p]
                minLoop=test_loop_cnt
            else:
                min_RMSE_model=[svdRMSE_int[i], svd_wo_p]
                minLoop=test_loop_cnt
            isThresUsed=False
            bestFitModel='SVDPP'
            
    print("\n\nRMSE values:")
    print("\n\nKNN: \nwith parameters - with threshold: {} without threshold: {}, \nwithout parameters - with threshold: {} without threshold: {}".format(knn_Arr['rmse']['with_param'][0],knn_Arr['rmse']['with_param'][1],knn_Arr['rmse']['without_param'][0],knn_Arr['rmse']['without_param'][1]))
    print("\n\nSVDPP: \nwith parameters - with threshold: {} without threshold: {}, \nwithout parameters - with threshold: {} without threshold: {}".format(svd_Arr['rmse']['with_param'][0],svd_Arr['rmse']['with_param'][1],svd_Arr['rmse']['without_param'][0],svd_Arr['rmse']['without_param'][1]))
    #print("\n\nTraining RMSE: SVDPP - {}, KNN - {}".format(svdRMSE_train[0], knRMSE_train[0]))
    print("\n\nAccuracy score:")
    print("\n\nKNN: \nwith parameters - with threshold: {} without threshold: {}, \nwithout parameters - with threshold: {} without threshold: {}".format(accuracy_score['knn']['with_param'][0],accuracy_score['knn']['with_param'][1],accuracy_score['knn']['without_param'][0],accuracy_score['knn']['without_param'][1]))
    print("\n\nSVDPP: \nwith parameters - with threshold: {} without threshold: {}, \nwithout parameters - with threshold: {} without threshold: {}".format(accuracy_score['svd']['with_param'][0],accuracy_score['svd']['with_param'][1],accuracy_score['svd']['without_param'][0],accuracy_score['svd']['without_param'][1]))
    
    print("\n\nKNN Threshold: {}".format(knn_thres))
    print("\n\nSVD Threshold: {}\n\n".format(svd_thres))

print("\nBest fit Model: {}, Minimum RMSE: {}, Threshold Model : {}, Loop number : {}, Is threshold values used: {}\n".format(bestFitModel, min_RMSE_model[0], min_thresArr, minLoop, isThresUsed))

#fill Matrix
fillMatrix(min_RMSE_model[1], isThresUsed)
    
#Generate output File
genFile(dataMat, 'output.txt', N, M)