###############################################################################

# Importing libraries

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re

from wordcloud import WordCloud
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.experimental import enable_hist_gradient_boosting

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

###############################################################################


data=pd.read_csv('../data/raw/mbti_1.csv')
# Remark :
    # The source data is a dataset file 'mbti_1.csv', whose relative path is '../data/raw/mbti_1.csv'
    # It is a public domain dataset collected from online posts.
    # This dataset contains over 8600 rows of data, on each row is a person’s:
        # 'type' : This persons 4 letter MBTI code/type
        # 'posts' : A section of each of the last 50 things they have posted (Each entry separated by "|||" (3 pipe characters))

data.info()
# Output :
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 8675 entries, 0 to 8674
    # Data columns (total 2 columns):
    #  #   Column  Non-Null Count  Dtype 
    # ---  ------  --------------  ----- 
    #  0   type    8675 non-null   object
    #  1   posts   8675 non-null   object
    # dtypes: object(2)
    # memory usage: 135.7+ KB

data.head()
# Output :
    # type	posts
    # 0	INFJ	'http://www.youtube.com/watch?v=qsXHcwe3krw|||...
    # 1	ENTP	'I'm finding the lack of me in these posts ver...
    # 2	INTP	'Good one _____ https://www.youtube.com/wat...
    # 3	INTJ	'Dear INTP, I enjoyed our conversation the o...
    # 4	ENTJ	'You're fired.|||That's another silly misconce...


###############################################################################

# Stratify split to ensure equal distribution of data

train_data,test_data=train_test_split(data,test_size=0.2,random_state=42,stratify=data.type)
print(train_data.shape, test_data.shape)
# Output :
    # (6940, 2) (1735, 2)


###############################################################################

def clean_text(text):
    cleaned_text=[]
    for sentence in tqdm(text):
        sentence=sentence.lower()
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        cleaned_text.append(sentence)
    return cleaned_text


# def lemmatize_text(text):
#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(word) for word in text.split() if len(word)>2]

# Building a preprocessing pipeline for text

preprocessing_pipeline = Pipeline([
    # ('clean_text', FunctionTransformer(clean_text, validate=False)),
    # ('lemmatize', FunctionTransformer(lemmatize_text, validate=False)),
    ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
])

# Fitting the preprocessing pipeline on the training text
preprocessing_pipeline.fit(train_data['posts'])

# Saving the pipeline
joblib.dump(preprocessing_pipeline, '../models/preprocessing_pipeline.joblib')

# Using the preprocessing pipeline to preprocess the data
train_post = preprocessing_pipeline.transform(train_data['posts'])
test_post = preprocessing_pipeline.transform(test_data['posts'])


###############################################################################
# Encoding the target variable

# Define the label encoder
target_encoder = LabelEncoder()

# Fit the encoder on the training data
target_encoder.fit(train_data.type)

# Save the encoder
joblib.dump(target_encoder, '../models/target_encoder.joblib')

# Use the encoder to preprocess the target
train_target = target_encoder.transform(train_data.type)
test_target = target_encoder.transform(test_data.type)



###############################################################################
# Model selection

models_accuracy={}

#------------------------------------------------------------------------------

# Logistic Regression

model_log=LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
model_log.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_log.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n',classification_report(test_target,model_log.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['logistic regression']=accuracy_score(test_target,model_log.predict(test_post))

#------------------------------------------------------------------------------

# Linear Support Vector Classifier

model_linear_svc=LinearSVC(C=0.1)
model_linear_svc.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_linear_svc.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n',classification_report(test_target,model_linear_svc.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['Linear Support Vector classifier']=accuracy_score(test_target,model_linear_svc.predict(test_post))

#------------------------------------------------------------------------------

# Support Vector Classifier

model_svc=SVC()
model_svc.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_svc.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_svc.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['Support Vector classifier']=accuracy_score(test_target,model_svc.predict(test_post))

#------------------------------------------------------------------------------

# Multinomial Naive Bayes

model_multinomial_nb=MultinomialNB()
model_multinomial_nb.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_multinomial_nb.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_multinomial_nb.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['Multinomial Naive Bayes']=accuracy_score(test_target,model_multinomial_nb.predict(test_post))

#------------------------------------------------------------------------------

# Decision Tree Classifier

model_tree=DecisionTreeClassifier(max_depth=14)
model_tree.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_tree.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_tree.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['Decision Tree classifier']=accuracy_score(test_target,model_tree.predict(test_post))

#------------------------------------------------------------------------------

# Random Forest Classifier

model_forest=RandomForestClassifier(max_depth=10)
model_forest.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_forest.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_forest.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['Random Forest Classifier']=accuracy_score(test_target,model_forest.predict(test_post))

#------------------------------------------------------------------------------

# XGBoost Classifier

# model_xgb=XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.1)
model_xgb=XGBClassifier(max_depth=5,n_estimators=50,learning_rate=0.1)
model_xgb.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_xgb.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_xgb.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['XGBoost Classifier']=accuracy_score(test_target,model_xgb.predict(test_post))

#------------------------------------------------------------------------------

# CatBoost Classifier

model_cat=CatBoostClassifier(loss_function='MultiClass',eval_metric='MultiClass',task_type='GPU',verbose=False)
model_cat.fit(train_post,train_target)

print('train classification report \n ',classification_report(train_target,model_cat.predict(train_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_cat.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

models_accuracy['CatBoost Classifier']=accuracy_score(test_target,model_cat.predict(test_post))

#------------------------------------------------------------------------------

# Accuracy comparizon

models_accuracy
models_accuracy.keys()
accuracy=pd.DataFrame(models_accuracy.items(),columns=['Models','Test accuracy'])
accuracy.sort_values(by='Test accuracy',ascending=False,ignore_index=True).style.background_gradient(cmap='Blues')


###############################################################################
# Exporting models

def export_models(models=[], path='../models/', active=False):
    """
    Export trained models
    - models : List of tuples (model, 'name')
    - path : directory where to save files
    - active : defines if the function should export models when called
    """
    if active :
        if not os.path.exists(path):
            os.makedirs(path)
        for model in models:
            file_path = path + model[1] + '.joblib'
            joblib.dump(model[0], file_path)
            print(f'Model saved to {file_path}')

models_list = [
    (model_log, "model_log"), # Logistic Regression
    (model_linear_svc, "model_linear_svc"), # Linear Support Vector Classifier
    (model_svc, "model_svc"), # Support Vector Classifier
    (model_multinomial_nb, "model_multinomial_nb"), # Multinomial Naive Bayes
    (model_tree, "model_tree"), # Decision Tree Classifier
    (model_forest, "model_forest"), # Random Forest Classifier
    (model_xgb, "model_xgb"), # XGBoost Classifier
    (model_cat, "model_cat"), # CatBoost Classifier
]

models_path = '../models/'

export_models(models=models_list, path=models_path, active=True)


# Exporting accuracy

# accuracy.to_pickle('../models_accuracy.pkl')
accuracy.to_csv('../results/models_accuracy.csv', index=False)









