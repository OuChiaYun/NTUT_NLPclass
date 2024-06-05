# 自然語言處理 HW3

# 組員 : 110590450 歐佳昀 110590452 莊于潔

110590450 歐佳昀(60%) : 作業架構、流程、分析、改善

110590452 莊于潔(40%) : 作業模型挑選、參數調整

## due 6/5,2024

- Goal: Named Entity Recognition on open datasets

- Input:

  - BTC NER dataset (IOB2)

- Output: Training a model to recognize the named entity types (to be detailed later)

### Tasks

- Performing NER on Twitter data (as detailed in the following slides)

- Data: an open dataset available from GitHub

- You have to submit the result of NER in terms of the F1 score

### Data:

- [Broad Twitter Corpus] available from GitHub
- Available at: https://gabrilovich.com/resources/data/wordsim353/wordsim353.html Two sets of word pairs with their similarity scores

- Format:
- 6 files in CoNLL format
- Each line contains: token ner_tag
- BIO or IOB format

以上資料集請放置於 data/

### ref:

- https://github.com/GateNLP/broad_twitter_corpus/blob/master/README.md (資料集切分)

## Result :

### 表現最好

============================================

\*CRF : \
Accuracy: 0.936 \
Weighted Average Precision: 0.927 \
Weighted Average Recall: 0.936 \
Weighted Average F1 Score: 0.928

============================================

---

### 其餘成果展示

- spacy

============================================

spaCy train - test: \
Accuracy:0.511\
Weighted Average Precision: 0.619\
Weighted Average Recall: 0.570\
Weighted Average F1 Score: 0.573

============================================

============================================

spaCy train - (test+dev): \
Accuracy:0.521\
Weighted Average Precision: 0.641 \
Weighted Average Recall: 0.587 \
Weighted Average F1 Score: 0.611

============================================

============================================

spaCy 5 k-fold : \
weight avg f1_score : 0.478 \
weight avg recall : 0.480 \
weightavg precision : 0.511 \
weightavg accuracy : 0.393

============================================

============================================

spaCy (train for command - dev for command - test): \
Accuracy: 0.781 \
Weighted Average Precision: 0.879 \
Weighted Average Recall: 0.825 \
Weighted Average F1 Score: 0.850

============================================

- NLTK

============================================

NLTK (train - test): \
Accuracy:0.891 \
Weighted Average Precision: 0.882 \
Weighted Average Recall: 0.891 \
Weighted Average F1 Score: 0.878

============================================

- implement \*CRF

============================================

\*CRF : \
Accuracy: 0.936 \
Weighted Average Precision: 0.927 \
Weighted Average Recall: 0.936 \
Weighted Average F1 Score: 0.928

============================================
