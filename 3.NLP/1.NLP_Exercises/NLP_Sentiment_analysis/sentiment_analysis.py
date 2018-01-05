#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
import os

os.chdir('D:/Data/DataScience/DeepLearning/Natural Language Processing/NLP_handson/training_data/')
sentiment_dict = {}
for each_line in open('sentiment_analysis.txt'):
    word,score = each_line.split('\t')
    sentiment_dict[word] = int(score)
    
user_comment = 'This is a good training and i appreciate it. But lab machines are worst'.lower().split()

print(user_comment)
print('over all user rating==>', 
      sum( sentiment_dict.get(word, 0) for word in user_comment ))

print(sentiment_dict.get('good', 0))
print(sentiment_dict.get('appreciate', 0))
print(sentiment_dict.get('worst', 0))

for word in user_comment:
    sentiment_dict.get(word,0)
    print(word, 'count', sentiment_dict.get(word,0))
    
