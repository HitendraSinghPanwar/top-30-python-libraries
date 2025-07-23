import gensim
import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, TfidfModel, LdaModel
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity

doc = open(r"C:\Users\thaku\Downloads\archive (1)\News_Category_Dataset_v3.json", encoding="utf-8")

tokenized = []
for sentence in doc.read().split('.'):
    tokenized.append(simple_preprocess(sentence, deacc=True))


w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4)
print("\n Similar Words to 'news':")
print(w2v_model.wv.most_similar("news", topn=5))


dictionary = Dictionary(tokenized)
corpus = [dictionary.doc2bow(text) for text in tokenized]
tfidf_model = TfidfModel(corpus)
print("\nTop TF-IDF words from 1st doc:")
for word_id, score in sorted(tfidf_model[corpus[0]], key=lambda x: -x[1])[:5]:
    print(dictionary[word_id], round(score, 4))

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=10)
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")


similarity_index = MatrixSimilarity(tfidf_model[corpus])
sim_scores = list(enumerate(similarity_index[tfidf_model[corpus[0]]]))
print("\nSimilarity of Doc[0] with others:")
for doc_id, score in sorted(sim_scores[1:6], key=lambda x: -x[1]):
    print(f"Doc {doc_id} similarity: {round(score, 3)}")