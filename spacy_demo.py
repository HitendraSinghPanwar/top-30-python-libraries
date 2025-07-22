#spaCy’s

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("spaCy is an industrial-strength NLP library built for production use.")

for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.dep_}\t{token.ent_type_}")

print("\nNamed Entity:")
for ent in doc.ents:
    print(f"{ent.text}\t{ent.label_}")

# displacy.serve(doc, style='dep', host='localhost', port=8080)
