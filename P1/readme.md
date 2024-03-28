# 自然語言處理 HW1

# 組員 : 110590450 歐佳昀 110590452 莊于潔

```
if data put in other_data or data folder, git will ignore to push it

please notice your necessary data in necessary_data folder

you can check wich folder ignore in .gitignore
```

## due 4/1,2024

- Goal: Sentiment classification on open source datasets

- Input: TSATC: Twitter Sentiment Analysis Training Corpus (to be detailed later)

- Output: Training classifiers to classify the sentiment of tweets (to be detailed later)

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
