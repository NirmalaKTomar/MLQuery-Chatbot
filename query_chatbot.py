# -*- coding: utf-8 -*-


pip install nltk

import nltk
import numpy as np
import random
import string 
import warnings
warnings.filterwarnings("ignore")

"""**DATASET :**

"""

#reading the text file
f=open('ml.txt','r',errors = 'ignore')
raw=f.read()
#convert to lowercase
raw = raw.lower()
print(raw)



nltk.download('punkt')
nltk.download('wordnet') 
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#sent_tokens is converting the data into a list of sentences
print(len(sent_tokens))
print(sent_tokens[:10])

#word_tokens is converting the sentences into the tokens
print(len(word_tokens))
print(word_tokens[:20])

#these are the preprocessing functions to return the normalized tokens

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

print(string.punctuation)

#some keywords for the greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#This is the main function that takes the user question and generates the chatbot response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    #initialize the model for tf-idf
    #inputs: Normalized tokens
    #output : bag of words    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    #find the cosine similarity of the user response with every other sentence in corpus
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0]
    #choose the best response
    i = -2
    while len(sent_tokens[idx[i]].split())<8:
      i-=1
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[i]    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx[i]]
        return robo_response

#the function that inputs the user response and output the chatbot response
flag=True
print("ROBO: My name is Robo. I will answer your queries about Machine learning. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                st = response(user_response)
                if st[0]=='[':
                  i=0
                  while st[i]!=']':
                    i+=1
                  print(st[i+1:])
                else:
                  print(st)
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")

