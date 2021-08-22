# -*- coding: utf-8 -*-
"""fasttext.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13HEEnADSbxkgjwvc0woZGlfD_uYHjFtW
"""

!wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
!unzip v0.9.2.zip

cd fastText-0.9.2

# for command line tool :
!make
# for python bindings :
!pip install .

!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz

!gzip -d cc.fa.300.bin.gz

cd /content/fastText-0.9.2

import fasttext.util

ft = fasttext.load_model('cc.fa.300.bin')

import pickle as pkl

all_words = pkl.load(open('/content/all_words_total.pkl', 'rb'))

embedding = {300:{}, 200:{}, 100:{}}
for word in all_words:
  embedding[300][word] = ft.get_word_vector(word)

pkl.dump(embedding[300], open('embedding_total_300dic.pkl', 'wb'))

files.download('embedding_total_300dic.pkl')

fasttext.util.reduce_model(ft, 100)