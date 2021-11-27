import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import dask.dataframe as dd
from dask.diagnostics import ProgressBar


from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_score, recall_score

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier




def load_features(path):
    ddf_features = dd.read_csv(path, sep='\t')

    print('загружаем df_features_short ...')
    with ProgressBar():
        df_features_short = ddf_features[['id', 'buy_time']].compute()

    print('сортируем df_features_short ...')
    df_features_short.sort_values(['buy_time'], inplace=True)
    df_features_short.rename(columns={'buy_time': 'features_buy_time'}, inplace=True)
    return (ddf_features, df_features_short)


def add_features(df, ddf_features, df_features_index):
    if 'features_buy_time' in df.columns and '0' in df.columns:
        print(f'Признаки уже добавлены, размер df {df.shape[0]} x {df.shape[1]}')
        return df

    print(f'Добавляем признаки в df {df.shape[0]} x {df.shape[1]} ...')
    
    print(f'сортируем df {df.shape[0]} x {df.shape[1]} ...')
    df = df.sort_values(['buy_time'])

    print('merge_asof ...')
    df_ft_keys = pd.merge_asof(
            df, 
            df_features_index,
            by='id', 
            left_on='buy_time', right_on='features_buy_time',
            direction='nearest')
    print(f'df_ft_keys: {df_ft_keys.shape[0]} x {df_ft_keys.shape[1]}')

    print('merge ...')
    with ProgressBar():
        ddf_ft = dd.merge(df_ft_keys, ddf_features, left_on=['id', 'features_buy_time'], right_on=['id', 'buy_time'])

    print('compute ...')
    with ProgressBar():
        df_ft = ddf_ft.compute().rename(columns={'buy_time_x': 'buy_time'})
        
    return df_ft


def train_val_split(df_train, val_portion=0.1):
    df_train_train = df_train[:int(len(df_train) * (1 - val_portion))]

    X_train_train = df_train_train.drop('target', axis=1)
    y_train_train = df_train_train['target']
    print(f'Тренировочный набор: {X_train_train.shape[0]} x {X_train_train.shape[1]}')

    df_train_val = df_train[int(len(df_train) * (1 - val_portion)):]
    X_train_val = df_train_val.drop('target', axis=1)
    y_train_val = df_train_val['target']
    print(f'Валидационный набор: {X_train_val.shape[0]} x {X_train_val.shape[1]}')

    return X_train_train, X_train_val, y_train_train, y_train_val

def report(y_true, y_proba):
    zero_division = 0
    metrics = np.array([[
                thr, 
                f1_score(y_true, y_proba > thr, zero_division=zero_division),
                precision_score(y_true, y_proba > thr, zero_division=zero_division),
                recall_score(y_true, y_proba > thr, zero_division=zero_division)
                ] for thr in np.linspace(0.00, 1.0, 50)])
    f1_threshold = metrics[np.argmax(metrics[:,1]), 0]
    y_pred = y_proba > f1_threshold
    print(f'y_pred = y_proba > {f1_threshold}')
    f1s = f1_score(y_true, y_pred, zero_division=zero_division)
    f1sm = f1_score(y_true, y_pred, average='macro', zero_division=zero_division)
    ras = roc_auc_score(y_true, y_proba)
    print('F1-score:', f1s)
    print('F1-score-macro:', f1sm)
    print('ROC-AUC-score:', ras)
    
    print(classification_report(y_true, y_pred, zero_division=zero_division))
    print(confusion_matrix(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr)
    plt.show()
    plt.plot(metrics[:,0], metrics[:,1], label='F1-score')
    plt.plot(metrics[:,0], metrics[:,2], label='precision')
    plt.plot(metrics[:,0], metrics[:,3], label='recall')
    plt.xlabel("Порог вероятности предсказания")
    plt.legend(loc="upper left")
    plt.show()
    return {'f1_threshold': f1_threshold, 'f1_score': f1s, 'f1_score_macro': f1sm, 'roc_auc_score': ras}



def column_selector(col_list):
    return ColumnTransformer([("selector", "passthrough", col_list)], remainder="drop")


def asstr(x):
    return x.astype(str)
    

class CallableWithArgs:
    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def __call__(self, fn_arg):
        return self.fn(fn_arg, *self.args)


def get_features(df):
    features = ['vas_id'] + [c for c in df.columns if c.isnumeric()]
    return features

def get_cat_features(df, cat_features):
    features = get_features(df)
    cat_features = [c for c in features if c in cat_features]
    return cat_features
    
def get_num_features(df, cat_features):
    features = get_features(df)
    cat_features = get_cat_features(df, cat_features)
    num_features = [c for c in features if c not in cat_features]
    return num_features

    
def make_feature_pipeline(cat_features):
    
    return Pipeline([
            ('prepare_features', FeatureUnion([
                ("categorical_features", make_pipeline(
                    column_selector(CallableWithArgs(get_cat_features, cat_features)),
                    SimpleImputer(strategy="most_frequent"),
                    FunctionTransformer(asstr),
                    OneHotEncoder(handle_unknown='ignore')
                )),
                ("numeric_features", make_pipeline(
                    column_selector(CallableWithArgs(get_num_features, cat_features)),
                    SimpleImputer(strategy="mean"),
                    StandardScaler()
                )),
            ])),
            # ухудшает результат
#            ('filter_important_features', SelectFromModel(
#                 LogisticRegression(penalty='l2', random_state=4242), 
#                 threshold=1e-3))
        ])


def spy(x):
    to_pickle(x, str(datetime.now().timestamp()) + ".spy.pkl")
    return x
    

def make_classifier(features_pipeline,  classifier_object):
    return Pipeline([ 
                ('features', features_pipeline),
#                ('spy', FunctionTransformer(spy)),
                ('classifier', classifier_object) 
            ])

def to_pickle(x, pickle_path):
    parent_path = os.path.dirname(pickle_path)
    if parent_path and parent_path != '.':
        os.makedirs(parent_path, exist_ok=True)
    print(f'Сохранение в  {pickle_path} ...')
    with open(pickle_path, 'wb') as out_strm:
        pickle.dump(x, out_strm)
    print(f'Сохранение в  {pickle_path} - ok')

def from_pickle(pickle_path):
    print(f'Загрузка из {pickle_path} ...')
    with open(pickle_path, 'rb') as in_strm:
        x = pickle.load(in_strm)    
    print(f'Загрузка из {pickle_path} - ok')
    return x

def iteration(classifier, df_train, model_path=None, cat_features=['vas_id'], val_portion=0.1):
    
    classifier_name = type(classifier).__name__
    print(f"Итерация с классификатором {classifier_name}, {len(cat_features)} категориальных признаков")
    
    print("Разделяем тренировочный набор ...")
    X_train_train, X_train_val, y_train_train, y_train_val = train_val_split(df_train, val_portion=val_portion)

    print(f"X_train_train: {X_train_train.shape[0]} x {X_train_train.shape[1]}")
    print(f"y_train_train: {y_train_train.shape[0]}")
    print(f"X_train_val: {X_train_val.shape[0]} x {X_train_val.shape[1]}")
    print(f"y_train_val: {y_train_val.shape[0]}")
    
    print("Создаем pipeline ...")
    feature_prep_pipeline = make_feature_pipeline(cat_features)
    pipeline = make_classifier(feature_prep_pipeline, classifier)

    print("Тренируем ...")
    pipeline.fit(X_train_train, y_train_train)

    if model_path:
        to_pickle(pipeline, model_path)

    metrics = {}

    if X_train_val.shape[0] > 0:
        if model_path:
            pipeline = from_pickle(model_path)

        print("Получаем предсказания ...")
        y_train_proba = pipeline.predict_proba(X_train_val)[:,1]

        print(" ------------------------ Отчет: ------------------------ ")
        metrics = report(y_train_val, y_train_proba)
    
    return {
        'pipeline': pipeline, 
        'classifier': classifier, 
        'cat_features': cat_features, 
        'metrics': metrics
    }

