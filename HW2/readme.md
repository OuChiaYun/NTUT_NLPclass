# 自然語言處理 HW1 Word Embeddings

# 組員 : 110590450 歐佳昀 110590452 莊于潔

110590450 歐佳昀
110590452 莊于潔

## due 4/29,2024

- Goal: Deriving word embeddings for estimating word similarity and analogy prediction on open datasets

- Input:

  - Word embeddings: fine-tuning pretrained models, or trained on your own
  - Text dataset

- Output: Result of word similarity and analogy prediction

### Tasks

- Deriving word embeddings for estimating word similarity and analogy prediction on open data (as detailed in the following slides)

- Data: an open dataset

- You have to submit the classification output

Example Word Embedding Models : Word2Vec、GloVe、fastText ..

### Data:

[WordSimilarity-353 Corpus] by Evgeniy Gabrilovich:

- Available at: https://gabrilovich.com/resources/data/wordsim353/wordsim353.html

  - Two sets of word pairs with their similarity scores

- [Bigger Analogy Test Set (BATS)] by Vecto team:

  - Available at: http://vecto.space/projects/BATS/
  - 98,000 questions in 40 morphological and semantic categories

### Format:

WordSim-353: Each set is available in two formats: CSV or Tab-delimited

    - The first two columns: word pairs
    - The third column: mean score for similarity

BATS: Word pairs with 40 different relations in 40 files

To train a classifier using the training set in any programming language
To test the classification result for the test set

point:

Tasks:

- (40pt) (1) Deriving a word embedding model
  Either fine-tuning a pretrained model
  Or training a new model
- (30pt) (2) Using word embedding for word similarity estimation
- (30pt) (3) Using word embedding for analogy prediction
  [Optional]:
- (25pt) (4) Compare with other document similarity estimation methods
  For example, co-occurrence matrix with TF-IDF, SVD, …
- (25pt) (5) Apply word embeddings in other tasks
  For example, classification, NER, …

### Output format:

Results

- Word embeddings
- Word similarity
- Correlation on WordSim353
- Analogy prediction
- Accuracy for BATS categories

other :

You can train your Word2Vec models using packages like genism

You can also implement your own codes using platforms like PyTorch, Keras, or TensorFlow

You can use the pretrained word embeddings from the following:

GloVe: https://nlp.stanford.edu/projects/glove/
Pretrained on Twitter, Wikipedia, …
Word2Vec: pretrained on Google News

The accuracy of many categories in BATS will be zero
Please focus on the categories with nonzero accuracies

## Report result

---

### Output format:

#### Task 1 :

word embeding :

- wrod2vec
- https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/README.md
- https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

- GLoVe
- https://github.com/stanfordnlp/GloVe
- https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip

- SVD
- https://github.com/valentinp72/svd2vec
- ```
   wget http://mattmahoney.net/dc/text8.zip -O text8.gz
   gzip -d text8.gz -f
  ```

#### Task 2 :

- (1)

  model name : word2vec

  diff_abs 平均值： 1.3917418687971976

  SignificanceResult(statistic=0.6845969668422078, pvalue=1.016190193647808e-49)

  => 強相關

- (2)

  model name : GLoVe

  diff_abs 平均值： 1.4636849824143916

  SignificanceResult(statistic=0.5987723194963509, pvalue=1.0213953289911825e-35)

  => 中等強相關(近強相關)

#### Task 3 :

(節選 1-3 表現最好者)

- (1)

  model name : GLoVe

  data/BATS\1_Inflectional_morphology: category [noun - plural_reg]\
  total accuracy - total: 35 ps: 35 lm: 35 data: 50\
  total: 0.7\
  Stemming: 0.7\
  Lemmatization: 0.7

  data/BATS\1_Inflectional_morphology: category [verb_ving - ved]\
  total accuracy - total: 38 ps: 41 lm: 38 data: 50\
  total: 0.76\
  Stemming: 0.82\
  Lemmatization: 0.76

  data/BATS\3_Encyclopedic_semantics: category [country - capital]\
  total accuracy - total: 46 ps: 43 lm: 46 data: 50\
  total: 0.92\
  Stemming: 0.86\
  Lemmatization: 0.92

- (2)

  model name : word2vec

  data/BATS\1_Inflectional_morphology: category [noun - plural_reg]\
  total accuracy - total: 32 ps: 35 lm: 32 data: 50\
  total: 0.64\
  Stemming: 0.7\
  Lemmatization: 0.64\

  data/BATS\1_Inflectional_morphology: category [verb_ving - ved]\
  total accuracy - total: 38 ps: 43 lm: 38 data: 50\
  total: 0.76\
  Stemming: 0.86\
  Lemmatization: 0.76

  data/BATS\1_Inflectional_morphology: category [verb_3psg - ved]\
  total accuracy - total: 27 ps: 30 lm: 27 data: 50\
  total: 0.54\
  Stemming: 0.6\
  Lemmatization: 0.54

- (3)

  model name : SVD - word2vec

  data/BATS\1_Inflectional_morphology: category [noun - plural_reg]\
  total accuracy - total: 11 ps: 11 lm: 11 data: 50\
  total: 0.22\
  Stemming: 0.22\
  Lemmatization: 0.22

  data/BATS\1_Inflectional_morphology: category [noun - plural_irreg]\
  total accuracy - total: 7 ps: 7 lm: 0 data: 50\
  total: 0.14\
  Stemming: 0.14\
  Lemmatization: 0.0

  data/BATS\1_Inflectional_morphology: category [verb_inf - 3psg]\
  total accuracy - total: 3 ps: 7 lm: 3 data: 50\
  total: 0.06\
  Stemming: 0.14\
  Lemmatization: 0.06
