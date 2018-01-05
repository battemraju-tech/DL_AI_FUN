#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
list_words = ["python","pythoner","pythoning","pythoned","pythonly","wolves", "leaves","horses","dogs","fairly",
              'dogs', 'leaves','wolves','babies','geese']
for word in list_words:
    print(ps.stem(word))
    

from nltk import stem
wn_lemat = stem.WordNetLemmatizer()
for word in list_words:
    print(word)
    print('steming==>', ps.stem(word))
    print('lemmati==>', wn_lemat.lemmatize(word))


#Conclusion:
    #Lemmatization have more accuracy than Stemming
    #Lemmatization gives almost Ditcionary or meaningful word
    #Stemming gives base form of word either it could be wrong or right.

 
