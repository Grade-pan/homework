import numpy as np
from sklearn import metrics, tree
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data():
    label = []
    #加载正样本
    A = np.zeros((6606, 1024), dtype=float)
    f = open('6606pos_5mer.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:1024]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    # 加载负样本
    B = np.zeros((6606, 1024), dtype=float)
    f = open('6606neg_5mer.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    B_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        B[B_row:] = list[0:1024]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        B_row += 1  # 然后方阵A的下一行接着读
    data = np.vstack((A,B))
    for i in range(0,6606):
        label.append(1)
    for i in range(0,6606):
        label.append(0)
    label = np.array(label)
    return data,label
#加载数据集进行分类
data,label = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=100)
def pingjia(y_pred,y_test,test_pro):
    print("ACC: %f " % accuracy_score(y_test, y_pred))
    print("AUC: %f" % metrics.roc_auc_score(y_test, test_pro[:, 1]))
    print("Precision: %f" % metrics.precision_score(y_test, y_pred))
    print("Recall: %f" % metrics.recall_score(y_test, y_pred))
    print("F1:: %f" % metrics.f1_score(y_test, y_pred))
'''
#KNN寻参   求得n_neighbors==15时最好
k_range=range(1,31)
k_scores=[]
for k in k_range:
    print(k)
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
plt.plot(k_range,k_scores)
plt.xlabel('The value of the k in KNN')
plt.ylabel('accuracy')
plt.show()
'''


#训练KNN模型
knn=KNeighborsClassifier(n_neighbors=15)
model = knn.fit(X_train,y_train)
y_pred = model.predict(X_test)
test_pro = model.predict_proba(X_test)
print("KNN:")
pingjia(y_pred,y_test,test_pro)

'''
#支持向量机寻参
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'gamma': [ 1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma': [ 1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]}
                    ]
scores = ['precision', 'recall']
svm = SVC()
grid_search = GridSearchCV(svm, tuned_parameters, cv=10,scoring="accuracy")
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
'''


#支持向量机训练
svm = SVC(C=1000,gamma=0.01,kernel='linear',probability=True)
model = svm.fit(X_train,y_train)
y_pred = model.predict(X_test)
test_pro = model.predict_proba(X_test)
print("SVM:")
pingjia(y_pred,y_test,test_pro)

'''
#Logistic回归寻参
parameters = {
    'penalty' : ['l1','l2'],
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
logreg = LogisticRegression()
clf = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=10)                     # number of folds
clf.fit(X_train,y_train)
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)
'''


#Logistic回归训练
logstic = LogisticRegression(C=1000.0,penalty='l2',solver='lbfgs')
model =logstic.fit(X_train,y_train)
y_pred = model.predict(X_test)
test_pro = model.predict_proba(X_test)
print("Logistic:")
pingjia(y_pred,y_test,test_pro)

'''
#随机森林寻参
# n_estimators=560,max_features=65,min_samples_split=30,max_depth=30
#{'n_estimators': range(200,701,50)}
#{'max_features':range(5,76,10)}{'min_samples_split':range(2,41,20)}
#{'max_features':range(5,76,10)}
#{'max_depth': range(24, 37, 3)}
rf_param_grid1 =  {'max_depth': range(24, 37, 3)}
rf_best = GridSearchCV(estimator=RandomForestClassifier(n_estimators=650,min_samples_split=2,max_features=65,max_depth=33),param_grid = rf_param_grid1,cv = 10,scoring="roc_auc",n_jobs= -1, verbose=100)
rf_best.fit(X_train,y_train)
print(rf_best.best_params_)
'''


#随机森林训练
RF = RandomForestClassifier(n_estimators=650,min_samples_split=2,max_features=65,max_depth=33)
model = RF.fit(X_train,y_train)
y_pred = model.predict(X_test)
test_pro = model.predict_proba(X_test)
print("RF:")
pingjia(y_pred,y_test,test_pro)



