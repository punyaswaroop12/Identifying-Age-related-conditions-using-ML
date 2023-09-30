!pip install tabpfn --no-index --find-links=file:///kaggle/input/pip-packages-icr/pip-packages
!mkdir -p /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff
!cp /kaggle/input/pip-packages-icr/pip-packages/prior_diff_real_checkpoint_n_0_epoch_100.cpkt /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff/

import numpy as np 
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
import optuna as opt
import lightgbm as lgb

import inspect
from collections import defaultdict
from tabpfn import TabPFNClassifier


from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')
# train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
# test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
# sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
# greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')


path = '/kaggle/input/icr-identify-age-related-conditions/'
df_train = pd.read_csv(path+'train.csv', index_col = 'Id')
df_test = pd.read_csv(path+'test.csv', index_col = 'Id')
greeks = pd.read_csv(path+'greeks.csv', index_col = 'Id')
sample_submission = pd.read_csv(path+'sample_submission.csv')
df = df_train.merge(greeks, how='left', left_index=True, right_index=True)

mask = df.Epsilon != 'Unknown'
df.loc[mask, 'Epsilon'] = df.loc[mask, 'Epsilon'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
df.loc[~mask, 'Epsilon'] = df.loc[mask, 'Epsilon'].median()
df['Epsilon'] = df['Epsilon'].astype(int)

def preprocessing(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['EJ'] = df['EJ'].replace({'A':0,'B':1}).astype(int)
    df.drop(columns=['CW', 'CF', 'GB', 'GE'], inplace=True)

    return df

train = preprocessing(df)
test = preprocessing(df_test)
test['Epsilon'] = train.Epsilon.max()

# y = train["Alpha"].replace({'A':0,'B':1,'D':2,'G':3, }).astype(int)
y = train["Class"]
X = train.drop(columns = ['Class', 'Alpha', 'Beta', 'Gamma', 'Delta'])
def balanced_log_loss(y_true, y_pred):
    # calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # calculate the weights for each class to balance classes
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    # calculate the predicted probabilities for each class
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    # calculate the summed log loss for each class
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    # calculate the weighted summed logarithmic loss
    # (factgor of 2 included to give same result as LL with balanced input)
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    # return the average log loss
    return balanced_log_loss/(N_0+N_1)


#     nc = np.bincount(ytrue);
#     return log_loss(ytrue, ypred, sample_weight = 1 / nc[ytrue], eps=1e-15);
class EnsembleAvgProba():
    
    def __init__(self, classfiers):
        
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.classifiers = classfiers
    
    def fit(self,X,y):
        
        y = y.values
        unique_classes, y = np.unique(y, return_inverse=True)
        self.classes_ = unique_classes
        first_category = X.EJ.unique()[0]
        X = self.imputer.fit_transform(X)
        for classifier in self.classifiers:
            if str(classifier).startswith('TabPFN'):
                classifier.fit(X,y,overwrite_warning =True)
            else :
                classifier.fit(X, y)
     
    def predict_proba(self, x):
        
        x = self.imputer.transform(x)
        probabilities = np.stack([classifier.predict_proba(x) for classifier in self.classifiers])
        averaged_probabilities = np.mean(probabilities, axis=0)
        class_0_est_instances = averaged_probabilities[:, 0].sum()
        others_est_instances = averaged_probabilities[:, 1:].sum()
        new_probabilities = averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) 
                                                                for i in range(averaged_probabilities.shape[1])]])
        
        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1) 


def training(model, x,y,y_meta):
    outer_results = list()
    best_loss = np.inf
    splits = 5
    cv_inner = KFold(n_splits=splits, shuffle=True, random_state=42)
    split = 0
    models=[]
    
    for train_idx,val_idx in tqdm(cv_inner.split(x), total = splits):
        split+=1
        x_train, x_val = x.iloc[train_idx],x.iloc[val_idx]
        y_train, y_val = y_meta.iloc[train_idx], y.iloc[val_idx]
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_val)
        probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
        p0 = probabilities[:,0]
        p0[p0 >= 0.5] = 1
        p0[p0 < 0.5] = 0
        y_p = 1- p0
        loss = balanced_log_loss(y_val,y_p)

        if loss<best_loss:
            best_model = model
            best_loss = loss
        outer_results.append(loss)
        print('>val_loss=%.5f, split = %.1f' % (loss,split))
    print('LOSS: %.5f' % (np.mean(outer_results)))
    return best_model, models
    
train_pred_and_time = X.copy()
test_predictors = y.copy()
ros = RandomOverSampler(random_state=42)
train_ros, y_ros = ros.fit_resample(X, y)

x_ros = train_ros
y_ = y_ros
params_lgb = {
    'boosting_type': 'GBDT',
    'random_state': 42,
    'verbose': -1
}

yt = EnsembleAvgProba(classfiers=[
                    XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.2,subsample=0.9,colsample_bytree=0.85),
#                     XGBClassifier(),
                    lgb.LGBMClassifier(**params_lgb),
                    TabPFNClassifier(N_ensemble_configurations=24),
                    TabPFNClassifier(N_ensemble_configurations=32)
                          ]
                )
Loading model that can be used for inference only
Using a Transformer with 25.82 M parameters
Loading model that can be used for inference only
Using a Transformer with 25.82 M parameters
m, models = training(yt,x_ros,y_,y_ros)

y_pred = m.predict_proba(test)
#y_pred_list = []
#for m in models:
#    y_pred_list.append(m.predict_proba(test_pred_and_time))
#y_pred=np.mean(y_pred_list, axis=0)
probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
p0 = probabilities[:,:1]
# p0[p0 > 0.60] = 1
# p0[p0 < 0.40] = 0
# p0 =  p.mean(axis=1) 
# p0[p0 > 0.86] = 1
# p0[p0 < 0.14] = 0


sample_submission['Id'] = test.reset_index()['Id']
sample_submission.class_0 = p0
sample_submission.class_1  = 1-p0
sample_submission.set_index('Id').to_csv('submission.csv')

# https://www.kaggle.com/code/renatoreggiani/icr-bestpublicscore-add-xgb-lgbm
