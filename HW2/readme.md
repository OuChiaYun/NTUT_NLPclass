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
