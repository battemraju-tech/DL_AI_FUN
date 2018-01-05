#Tokenizing
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
words = word_tokenize("We are learning Artificial Intelligence")
print (words)
#PartOfSpeechTagging
pos_words = pos_tag(words)
print (pos_words)



#Chunking
def process_content():
    try:
        for i in words:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(word)
            print(tagged)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()     
            
    except Exception as e:
        print(str(e))


process_content()


#Chinking
def process_content():
    try:
        for i in words:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(word)
            print(tagged)
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()     
            
    except Exception as e:
        print(str(e))


process_content()
