import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc    

data = pd.read_csv("data/train_150k.txt", header=None, names=["target", "text"],sep='\t')
data_test = pd.read_csv("data/test_62k.txt", header=None, names=["target", "text"],sep='\t')

data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]
dataset = pd.concat([data_pos, data_neg])


dataset['text']=dataset['text'].str.lower()

def cleaning_URLs(data):
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',' ',data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))
data_test['text'] = data_test['text'].apply(lambda x: cleaning_URLs(x))

import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))
data_test['text']= data_test['text'].apply(lambda x: cleaning_punctuations(x))

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1\1', text)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
data_test['text'] = data_test['text'].apply(lambda x: cleaning_repeating_char(x))

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
data_test['text'] = data_test['text'].apply(lambda x: cleaning_numbers(x))

def cleaning_single_c(data):
    return re.sub(r'\b[a-zA-Z]\b', '', data)

dataset['text'] = dataset['text'].apply(lambda x: cleaning_single_c(x))
data_test['text'] = data_test['text'].apply(lambda x: cleaning_single_c(x))

def lemmatize_text(text):
    # init WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # 將文本分詞並將每個單詞轉換為其原型形式
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    # 將單詞列表重新組合為文本
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text
dataset['text'] = dataset['text'].apply(lambda x: lemmatize_text(x))
data_test['text'] = data_test['text'].apply(lambda x: lemmatize_text(x))

def cleaning_non_eng(data):
    cleaned_data = re.sub(r'[^a-zA-Z\s]', '', data)
    return cleaned_data
dataset['text'] = dataset['text'].apply(cleaning_non_eng)
data_test['text'] = data_test['text'].apply(cleaning_non_eng)

def cleaning_multi_space(data):
    return re.sub(r'\s+', ' ', data, flags=re.I)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_multi_space(x))
data_test['text'] = data_test['text'].apply(lambda x: cleaning_multi_space(x))

dataset.to_csv("other_data/dataset.csv",index=False)

stopwordlist = ['a','an','about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves','today','day','½t','im','go','<UNK>']
            
            #  'today','day','im','i m','go','get','got','time','morning','tomorrow','amp',###
            #  'going','really','one','twitter','wa','like','ill','½s','thats','still','but',
            #  'know','½t','make','see','ive','much','off']  ###
            
            
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

data_neg = dataset['text'][74968:]
wc_neg = WordCloud(stopwords=stopwordlist ,max_words=2000, width=1600, height=800, collocations=False).generate(" ".join(data_neg))
axes[0].imshow(wc_neg, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Negative Words')  # 使用 set_title() 方法设置标题

data_pos = dataset['text'][:74968]
wc_pos = WordCloud(stopwords=stopwordlist ,max_words=2000, width=1600, height=800, collocations=False).generate(" ".join(data_pos))
axes[1].imshow(wc_pos, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Positive Words')  # 使用 set_title() 方法设置标题

plt.show()


dataset = dataset.sample(frac=1).reset_index(drop=True) #將訓練集打亂

X_train = dataset.text
y_train = dataset.target
X_test = data_test.text
y_test = data_test.target

vectoriser = TfidfVectorizer( min_df=7,stop_words=stopwordlist,max_features=3500)

vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


feature_names_df = pd.DataFrame({'Feature Names': vectoriser.get_feature_names_out()})
# 將所選特徵保存為CSV文件
feature_names_df.to_csv('other_data/feature_names.csv', index=False)

def model_Evaluate(model,name):
    # Predict values for Test dataset
    if(name =="GaussianNB"):
        
        y_pred = model.predict(X_test.toarray())  # Convert to array
    else:
        y_pred = model.predict(X_test)
    print(name)
    print(classification_report(y_test, y_pred))
    report  = classification_report(y_test, y_pred,output_dict=True)
    
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f_measure = report['weighted avg']['f1-score']
    accuracy = report['accuracy']

    
    print("Precision: {:.2f}". format(precision))
    print("Recall: {:.2f}". format(recall))    
    print("F-measure: {:.2f}". format(f_measure))
    print("Accuracy: {:.2f}\n". format(accuracy))
    
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)

# Train the model
logreg_model.fit(X_train, y_train)

# Evaluate the model
model_Evaluate(logreg_model, "LogisticRegression")

from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes model
gnb_model = GaussianNB()

# Train the model
gnb_model.fit(X_train.toarray(), y_train)  # GaussianNB requires array input

# Evaluate the model
model_Evaluate(gnb_model, "GaussianNB")


BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel,'BernoulliNB')

MNB_model = MultinomialNB()
MNB_model.fit(X_train, y_train)
model_Evaluate(MNB_model,"MultinomialNB")


X_train = dataset.text
y_train = dataset.target
X_test = data_test.text
y_test = data_test.target

vectorizer = CountVectorizer(stop_words=stopwordlist,max_features=3500)

text = vectorizer.fit_transform(dataset['text'])
X_test  = vectorizer.transform(data_test['text'])

feature_names_df = pd.DataFrame({'Feature Names': vectorizer.get_feature_names_out()})
# 將所選特徵保存為CSV文件
feature_names_df.to_csv('other_data/feature_names_CountVectorizer.csv', index=False)


MNB_model = MultinomialNB()

MNB_model.fit(text, y_train)
model_Evaluate(MNB_model,"MNB_model")

BNBmodel = BernoulliNB()

BNBmodel.fit(text, y_train)
model_Evaluate(BNBmodel,"BNBmodel")

from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(max_iter=200)

logreg_model.fit(text, y_train)

model_Evaluate(logreg_model, "LogisticRegression")
