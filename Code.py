import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
def score(y_true,y_pred,t=0.5):
    y_pred_t = np.where(y_pred>t,1,0)
    p = precision_score(y_true,y_pred_t)
    r = recall_score(y_true,y_pred_t)
    f = f1_score(y_true,y_pred_t)
    acc = accuracy_score(y_true,y_pred_t)
    auc = roc_auc_score(y_true,y_pred)
    #print('precision:{:.4f} recall:{:.4f} f1:{:.4f} accuracy:{:.4f}'.format(p,r,f,acc))
    return acc,p,r,f,auc

data = pd.read_excel("缩尾前数据.xls")
print(data.info())

from sklearn.preprocessing import MinMaxScaler

X = data.values[:, 2:-1]
y = data.values[:, -1].astype(int)
y = y - 1
X = MinMaxScaler().fit_transform(X)


print(set(y))

n_folds = 10

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

result = pd.DataFrame(columns = ['precision','recall','f1','accuracy','auc','acc1','acc2'])
def get_result(estimators,random_state=2023,save_name=''):
    print('=================={}=================='.format(save_name))
    metrics = []
    y_pred = pd.Series(0,index=range(len(y)))
    for j,estimator in enumerate(estimators):
        kf = StratifiedKFold(n_splits=10,random_state=random_state+j,shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(X,y=y)):
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            estimator.fit(train_X, train_y)
            pre = estimator.predict_proba(test_X)[:,1]
            metrics.append([score(test_y,pre)])
            y_pred.loc[test_index] += pre/len(estimators)
    acc,p,r,f,auc = score(y,y_pred.values)
    print('precision:{:.4f} recall:{:.4f} f1:{:.4f} accuracy:{:.4f} auc:{:.4f}'.format(p,r,f,acc,auc))
    #Confusion Matrix
    cm = confusion_matrix(y, np.where(y_pred>0.5,1,0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm
                                  )
    disp.plot(values_format='d', cmap=plt.cm.Blues)
    plt.savefig(save_name+'cm.png', dpi=600, bbox_inches='tight')
    plt.show()
    #Auc
    fpr, tpr, thresholds = roc_curve(y, y_pred.values)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot() 
    plt.savefig(save_name+'auc.png',dpi = 600,bbox_inches = 'tight')
    plt.show()
    acc_1 = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0
    acc_2 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 0] + cm[1, 1] > 0 else 0
    result.loc[save_name] = [p,r,f,acc,auc,acc_1,acc_2]
    return ''#np.mean(metrics, axis=0)

from sklearn.svm import SVC

svm_list = [SVC(random_state=2023+_, C=20, gamma=0.1,probability=True) for _ in range(n_folds)]
print(get_result(svm_list,save_name='SVC'))

from sklearn.linear_model import LogisticRegression

lr_list = [LogisticRegression(n_jobs=-1, random_state=2023+_, C=10000) for _ in range(n_folds)]
print(get_result(lr_list,save_name='LogisticRegression1'))

from sklearn.linear_model import LogisticRegression

lr_list = [LogisticRegression(n_jobs=1, random_state=2023+_, C=10, penalty="l2", solver="liblinear") for _ in
           range(n_folds)]  # Ridge regression
print(get_result(lr_list,save_name='LogisticRegression2'))

from sklearn.linear_model import LogisticRegression

lr_list = [LogisticRegression(n_jobs=1, random_state=2023+_, C=10, penalty="l1", solver="liblinear") for _ in
           range(n_folds)]  # Lasso regression
print(get_result(lr_list,save_name='LogisticRegression3'))



from sklearn.ensemble import GradientBoostingClassifier

gb_list = [
    GradientBoostingClassifier(random_state=2023+_, n_estimators=100, learning_rate=0.2, subsample=0.9, max_depth=10,
                               verbose=0) for _ in range(n_folds)]
print(get_result(gb_list,save_name='GradientBoostingClassifier'))

from xgboost import XGBClassifier

xgb_list = [
    XGBClassifier(random_state=2023+_, n_estimators=150, learning_rate=0.2, subsample=0.9, max_depth=20, n_jobs=-1,
                  use_label_encoder=False) for _ in range(n_folds)]
print(get_result(xgb_list,save_name='XGBClassifier'))

from lightgbm import LGBMClassifier

lgbm_list = [
    LGBMClassifier(random_state=2023+_, n_estimators=150, learning_rate=0.2, subsample=0.9, max_depth=20, n_jobs=-1) for _
    in range(n_folds)]
print(get_result(lgbm_list,save_name='LGBMClassifier'))

from sklearn.ensemble import AdaBoostClassifier

ada_list = [AdaBoostClassifier(random_state=2023+_, n_estimators=150, learning_rate=0.2) for _ in range(n_folds)]
print(get_result(ada_list,save_name='AdaBoostClassifier'))



from sklearn.neural_network import MLPClassifier

mlp_list = [MLPClassifier(random_state=2023+_, hidden_layer_sizes=[10,10], learning_rate_init=0.01, max_iter=400) for _
            in range(n_folds)]
print(get_result(mlp_list,save_name='MLPClassifier'))

from sklearn.tree import DecisionTreeClassifier

dt_list = [DecisionTreeClassifier(random_state=2023+_, max_depth=20) for _ in range(n_folds)]
print(get_result(dt_list,save_name='DecisionTreeClassifier'))

from sklearn.ensemble import RandomForestClassifier

rf_list = [RandomForestClassifier(random_state=2023+_, max_depth=20, n_estimators=100) for _ in range(n_folds)]
print(get_result(rf_list,save_name='RandomForestClassifier'))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bg_list = [BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=20, random_state=2023+_), n_estimators=200,
                             random_state=2023+_, max_features=5) for _ in range(n_folds)]
print(get_result(bg_list,save_name='BaggingClassifier'))

from catboost import CatBoostClassifier

cat_list = [
    CatBoostClassifier(random_state=2023+_, n_estimators=100, learning_rate=0.2, subsample=0.9, max_depth=10, verbose=0)
    for _ in range(n_folds)]
print(get_result(cat_list,save_name='CatBoostClassifier'))

result.to_excel('result.xlsx')