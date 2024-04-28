from gensim.models.keyedvectors import Word2VecKeyedVectors
from svd2vec import svd2vec 
# https://github.com/valentinp72/svd2vec

### cmd ###
# pip install svd2vec
# wget http://mattmahoney.net/dc/text8.zip -O text8.gz
# gzip -d text8.gz -f
###

documents = [open("text8", "r").read().split(" ")]
svd = svd2vec(documents, window=2, min_count=100)

svd.save_word2vec_format("model/svd_word2vec_format.txt")