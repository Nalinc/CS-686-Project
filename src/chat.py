import os
from scipy import spatial
import numpy as np
import gensim
import nltk
import sys
from keras.models import load_model


#import theano
#theano.config.optimizer="None"
if(len(sys.argv)!=2):
    print("specify path to word2vec.bin folder")
    sys.exit()
else:
    path = sys.argv[1]
    if (path[-1]) != "/":
        path+="/"


model=load_model('./models/LSTM5000.h5')
#os.chdir("/home/nc/corpus/apnews_sg");
mod = gensim.models.Word2Vec.load(path+'word2vec.bin');
while(True):
    x=input("Enter the message:");
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    output=' '.join(outputlist)
    print(output)
