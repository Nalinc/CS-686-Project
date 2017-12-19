import sys
import os
import pickle
import numpy as np
from keras.models import Sequential
import gensim
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import theano
# theano.config.optimizer="None"

with open('conversation.pickle','rb') as f:
    vec_x,vec_y=pickle.load(f)    
    
vec_x=np.array(vec_x,dtype=np.float64)
vec_y=np.array(vec_y,dtype=np.float64)    

x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)

model=Sequential()
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM500.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM1000.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM1500.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM2000.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM2500.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM3000.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM3500.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM4000.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM4500.h5');
# model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
# model.save('LSTM5000.h5');

# model = load_model('LSTM4500.h5')

history = model.fit(x_train, y_train, nb_epoch=5000,validation_data=(x_test, y_test))

# with open('history.pickle','wb') as f:
#     pickle.dump(history,f)

model.save('LSTM5000_bulk.h5')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


predictions=model.predict(x_test) 
#os.chdir("/home/nc/corpus/apnews_sg")
if(len(sys.argv)!=2):
    print("specify path to word2vec.bin folder")
    sys.exit()
else:
    path = sys.argv[1]
    if (path[-1]) != "/":
        path+="/"
mod = gensim.models.Word2Vec.load(path+'word2vec.bin');   
[mod.most_similar([predictions[10][i]])[0] for i in range(15)]
