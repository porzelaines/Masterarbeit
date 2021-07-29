import os
import json
import nltk as nlp
import re
nlp.download('punkt')
nlp.download('stopwords')
counter_tweets = 0    
len_tweets = 0    
len_characters = 0
document=''
default_stopwords = set(nlp.corpus.stopwords.words('english'))
for subdir, dirs, files in os.walk(r'./Testsets/Depression'):
    for file in files:        
        profile = open(r'./Testsets/Depression/%s' %file,
                          encoding='utf8')
        profile = json.load(profile)
        for k,v in profile.items():
            document += v['0'] + ' '
            
document= re.sub(r'[^\w]', ' ', document)
document=document.lower()
tokens=nlp.word_tokenize(document)
types=nlp.Counter(tokens)
TTR= (len(types)/len(tokens))*100
words = [word for word in tokens if word not in default_stopwords]
fdist = nlp.FreqDist(words)
for word, frequency in fdist.most_common(13):
    print(u'{};{}'.format(word, frequency))