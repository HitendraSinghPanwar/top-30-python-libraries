import gensim
import os
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.test.utils import get_tmpfile

doc = open(r"C:\Users\thaku\Downloads\archive (1)\News_Category_Dataset_v3.json", encoding ='utf-8')
tokenized =[]
for sentence in doc.read().split('.'):
  tokenized.append(simple_preprocess(sentence, deacc = True))
print(tokenized)

my_dictionary = corpora.Dictionary(tokenized)

my_dictionary.save('my_dictionary.dict')
load_dict = corpora.Dictionary.load('my_dictionary.dict')

tmp_fname = get_tmpfile("dictionary")
my_dictionary.save_as_text(tmp_fname)
load_text = corpora.Dictionary.load_from_text(tmp_fname)

BoW_corpus =[my_dictionary.doc2bow(doc, allow_update = True) for doc in tokenized]
print(BoW_corpus)