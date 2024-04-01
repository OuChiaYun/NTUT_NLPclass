# 自然語言處理 HW1

# 組員 : 110590450 歐佳昀 110590452 莊于潔

```
if data put in other_data or data folder, git will ignore to push it

please notice your necessary data in necessary_data folder

you can check wich folder ignore in .gitignore
```

## due 4/1,2024

- Goal: Sentiment classification on open source datasets

- Input: TSATC: Twitter Sentiment Analysis Training Corpus

- Output: Training classifiers to classify the sentiment of tweets

### Tasks

- Performing sentiment classification on twitter data (as detailed in the following slides)

- Data: an open dataset from Huggingface

- You have to submit the classification output

### Data:

[TSATC: Twitter Sentiment Analysis Training Corpus] from Hugging Face
1,578,627 tweets, about 15MB in size
Available at:
https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis

Format:
Two text files consisting of lines of records
Each record contains 2 columns: feeling, text

To train a classifier using the training set in any programming language
To test the classification result for the test set

### Output format:

Classification results

- Precision
- Recall
- F-measure
- Accuracy

## Report - best model :

### Peformance

Classification results

- Precision = 0.77
- Recall = 0.77
- F-measure = 0.77
- Accuracy = 0.77

\*\*因為最好結果所有印出來的都是 0.77，故此處寫 0.77

##### Logistic Regression with TfidfVectorizer

|              | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | -------- | ------- |
|      0       |   0.78    |  0.76  | 0.77     | 30969   |
|      1       |   0.76    |  0.79  | 0.78     | 31029   |
|   accuracy   |           |        | 0.77     | 61998   |
|  macro avg   |   0.77    |  0.77  | 0.77     | 61998   |
| weighted avg |   0.77    |  0.77  | 0.77     | 61998   |

##### BernoulliNB with TfidfVectorizer

|              | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | -------- | ------- |
|      0       |   0.77    |  0.75  | 0.76     | 30969   |
|      1       |   0.76    |  0.78  | 0.77     | 31029   |
|   accuracy   |           |        | 0.77     | 61998   |
|  macro avg   |   0.77    |  0.77  | 0.77     | 61998   |
| weighted avg |   0.77    |  0.77  | 0.77     | 61998   |

##### Logistic Regression with CountVectorizer

|              | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | -------- | ------- |
|      0       |   0.75    |  0.75  | 0.77     | 30969   |
|      1       |   0.80    |  0.80  | 0.78     | 31029   |
|   accuracy   |           |        | 0.77     | 61998   |
|  macro avg   |   0.77    |  0.77  | 0.77     | 61998   |
| weighted avg |   0.77    |  0.77  | 0.77     | 61998   |

##### BernoulliNB Regression with CountVectorizer

|              | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | -------- | ------- |
|      0       |   0.77    |  0.75  | 0.76     | 30969   |
|      1       |   0.76    |  0.78  | 0.77     | 31029   |
|   accuracy   |           |        | 0.77     | 61998   |
|  macro avg   |   0.77    |  0.77  | 0.77     | 61998   |
| weighted avg |   0.77    |  0.77  | 0.77     | 61998   |

<br>
<br>

### The list we try and use :

#### Data preprocessing way :

|                    Way                     | Expected Use |    Final Use    |
| :----------------------------------------: | :----------: | :-------------: |
|         Convert text to lowercase:         |      v       |        v        |
|               Cleaning URLs                |      v       |        v        |
|    Removing punctuation and odd symbols    |      v       |        v        |
| Replacing consecutive repeating characters |      v       |        v        |
|              Cleaning numbers              |      v       |        v        |
|         Cleaning single characters         |      v       |        v        |
|             Lemmatizing words              |      v       |        v        |
|         Cleaning non-English words         |      v       |        v        |
|           Cleaning extra spaces            |      v       |        v        |
|          Word Cloud Visualization          |      v       |        v        |
|              Set Unknown Word              |      v       | x too long time |

#### Text feature extraction methods :

|       Way       | Expected Use | Final Use |
| :-------------: | :----------: | :-------: |
| TfidfVectorizer |      v       |     v     |
| CountVectorizer |      v       |     v     |

#### model:

|           Way           | Expected Use |    Final Use    |
| :---------------------: | :----------: | :-------------: |
|   Logistic Regression   |      v       |        v        |
|  Gaussian Naive Bayes   |      v       |        v        |
|  Bernoulli Naive Bayes  |      v       |        v        |
| Multinomial Naive Bayes |      v       |        v        |
|           SVM           |      v       | x too long time |
|   k-Nearest Neighbors   |      v       | x too long time |

extra knowledge we use : word cloud、CountVectorizer
