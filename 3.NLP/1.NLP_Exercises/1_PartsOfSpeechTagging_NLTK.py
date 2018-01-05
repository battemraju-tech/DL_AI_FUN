#This means labeling words in a sentence as nouns, adjectives, 
#verbs...etc. Even more impressive, it also labels by tense, and more. 
#Here's a list of the tags, what they mean, and some examples:

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


#One is a State of the Union address from 2005, 
#and the other is from 2006 from past President George W. Bush.
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()

