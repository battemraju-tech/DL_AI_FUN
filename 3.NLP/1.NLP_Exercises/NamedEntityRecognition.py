import nltk
from nltk.corpus import state_union
import nltk.tokenize 

sample_text = "Steve Jobs was the CEO of Apple Corp."
sample_text = sample_text.split()
#tokenized = nltk.tokenize(sample_text)

tagged = nltk.pos_tag(sample_text)
namedEnt = nltk.ne_chunk(tagged, binary=True)
namedEnt.draw()

def process_content():
    try:
        for i in sample_text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            print(namedEnt)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()