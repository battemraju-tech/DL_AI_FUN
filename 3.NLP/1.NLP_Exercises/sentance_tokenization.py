from nltk.tokenize import sent_tokenize
sentence = sent_tokenize("Hello NLP learner. This is an NLP python script")
print (sentence)

story1 = 'Hi, This is Raju. I am data scientist. '
story2 = 'I love to work on machine learing.'
story3 = 'I love to learn data science technologies'
story1=story1+story2+story3
story = sent_tokenize(story1) 
print(story)