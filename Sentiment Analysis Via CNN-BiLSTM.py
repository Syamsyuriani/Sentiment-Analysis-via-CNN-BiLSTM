## Import required modules
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import text
import matplotlib.pyplot as plt

## Upload Data
quipper_df = pd.read_csv('https://raw.githubusercontent.com/Syamsyuriani/Scrapping_Data/main/Quipper-Data_labelled.csv')
quipper_df.columns
review = quipper_df.drop(columns=['Unnamed: 2', 'Unnamed: 3'])

## Remove duplicates
dt = review.drop_duplicates(subset=['review'], keep='last', inplace=False).reset_index()
df = dt.drop(['index'], axis=1)

## Calculates and displays positive and negative data graphs
negative = df[df['label']==-1].count()[0]
positive = df[df['label']==1].count()[0]

labels = ['Positive','Negative']
Category1 = [positive, negative]
plt.bar(labels, Category1, tick_label=labels, width=0.5, color=['coral', 'c'])
plt.xlabel('Sentiment Class')
plt.ylabel('Data')
plt.title('Sentiment Analysis Data Bar Chart')

## CLEANING DATA
stopwords = pd.read_csv("https://raw.githubusercontent.com/listakurniawati/COVID-19-With-SVM/main/stopwords_id.csv?token=ARCQD7EZ55J4TUTAWYYLYOTAX3FTW")
stopwords = np.append(stopwords, "rt")
 
def clean_text(tweet):
 
    # Convert to lower case
    tweet = tweet.lower()
    # Clean www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    # Clean @username
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove punctuation
    tweet = re.sub(r'[^\w\s]',' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove punctuation
    tweet = re.sub(r'[^\w\s]',' ', tweet)
    #Clean number
    tweet = re.sub(r'[\d-]', '', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #trim
    tweet = tweet.strip('\'"')
    # Clean per Words
    words = tweet.split()
    tokens=[]
    for ww in words:
        #split repeated word
        for w in re.split(r'[-/\s]\s*', ww):
            #replace two or more with two occurrences
            pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
            w = pattern.sub(r"\1\1", w)
            #strip punctuation
            w = w.strip('\'"?,.')
            #check if the word cosists of two or more alphabets
            val = re.search(r"^[a-zA-Z][a-zA-Z][a-zA-Z]*$", w)
            #add tokens
            if(w in stopwords or val is None):
                continue
            else:
                tokens.append(w.lower())
    
    tweet = " ".join(tokens)
    return tweet
 
df['label'] = df['label'].replace(-1,0)
df['review'] = df['review'].map(lambda x: clean_text(x))
df = df[df['review'].apply(lambda x: len(x.split()) >=1)]

## Tokenization
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['review'])

## Sequencing
sequences = tokenizer.texts_to_sequences(df['review'])
x = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=80)
y = np.array((df['label']))

## Dataset Split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

seq_length = x.shape[1]
vocab_size = len(tokenizer.index_word) + 1

## CNN-BiLSTM Models
pip install -q -U keras-tuner

from tensorflow import keras
from kerastuner.tuners import BayesianOptimization
 
def cnn_bilstm(hp):

    #Input layer
    inputs = keras.layers.Input(shape=(seq_length,))
 
    #Embedding
    embedding = keras.layers.Embedding(input_dim = vocab_size,
                                       output_dim = hp.Choice('embedding_size', values = [64, 100, 128]))(inputs)
 
    #Convolution layer
    ngram_1 = keras.layers.Conv1D(filters = hp.Int('filters1',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                                  kernel_size = hp.Int('kernel_size1',
                                                min_value = 3, 
                                                max_value = 5, 
                                                step = 1),
                                  activation='relu',
                                  kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn1',
                                                                                       values = [0.0, 0.01])))(embedding)
    ngram_2 = keras.layers.Conv1D(filters = hp.Int('filters2',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                                  kernel_size = hp.Int('kernel_size2',
                                                min_value = 3, 
                                                max_value = 5, 
                                                step = 1),
                                  activation='relu',
                                  kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn2',
                                                                                       values = [0.0, 0.01])))(embedding)
    ngram_3 = keras.layers.Conv1D(filters = hp.Int('filters3',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                                  kernel_size = hp.Int('kernel_size3',
                                                min_value = 3, 
                                                max_value = 5, 
                                                step = 1),
                                  activation='relu',
                                  kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn3',
                                                                                       values = [0.0, 0.01])))(embedding)
    ngram_4 = keras.layers.Conv1D(filters = hp.Int('filters4',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                                  kernel_size = hp.Int('kernel_size4',
                                                min_value = 3, 
                                                max_value = 5, 
                                                step = 1),
                                  activation='relu',
                                  kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn4',
                                                                                       values = [0.0, 0.01])))(embedding)
 
    #Max Pooling layer
    ngram_1 = keras.layers.GlobalMaxPooling1D()(ngram_1)
    ngram_2 = keras.layers.GlobalMaxPooling1D()(ngram_2)
    ngram_3 = keras.layers.GlobalMaxPooling1D()(ngram_3)
    ngram_4 = keras.layers.GlobalMaxPooling1D()(ngram_4)
    merged = keras.layers.Concatenate(axis=1)([ngram_1, ngram_2, ngram_3, ngram_4])
 
    #BiLSTM layer
    bilstm1 = keras.layers.Layer(LSTM(units = hp.Int('units1',
                                                     min_value = 100,
                                                     max_value = 200,
                                                     step = 50),
                                      kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_regularizer1',
                                                                                         values = [0.0, 0.01])),
                                      recurrent_regularizer=keras.regularizers.l2(hp.Choice('rec_regularizer1',
                                                                                            values = [0.0, 0.01])),
                                      return_sequences = True))(merged)
    bilstm2 = keras.layers.Layer(LSTM(units = hp.Int('units2',
                                                     min_value = 100,
                                                     max_value = 200,
                                                     step = 50),
                                      kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_regularizer2',
                                                                                         values = [0.0, 0.01])),
                                      recurrent_regularizer=keras.regularizers.l2(hp.Choice('rec_regularizer2',
                                                                                            values = [0.0, 0.01])),
                                      return_sequences = True, go_backwards=True))(bilstm1)
    
    #Dropout layer
    lstm_out = keras.layers.Dropout(0.25)(bilstm2)
 
    #Output layer
    output = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_dense', values = [0.0, 0.01])))(lstm_out)
    model = keras.models.Model(inputs=inputs, outputs=output)
 
    model.compile(optimizer = keras.optimizers.Adam(
                              hp.Choice('learning_rate', 
                                        values = [1e-2, 1e-3, 1e-4])),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
 
    return model
 
tuner = BayesianOptimization(cnn_bilstm,
                             objective = 'val_accuracy', 
                             max_trials = 10,
                             directory = 'Result',
                             project_name = 'Sentiment_CNN-BiLSTM')
 
tuner.search(x_train, y_train, batch_size=32, epochs = 10, validation_data = (x_test, y_test))
 
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
 
print('\nThe hyperparameter search is complete. \nembedding_size:', best_hps.get('embedding_size'),
      '\nfilters:', [best_hps.get('filters1'), best_hps.get('filters2'), best_hps.get('filters3'), best_hps.get('filters4')],
      '\nkernel_size:', [best_hps.get('kernel_size1'), best_hps.get('kernel_size2'), best_hps.get('kernel_size3'), best_hps.get('kernel_size4')],
      '\nkernel_cnn:', [best_hps.get('kernel_cnn1'), best_hps.get('kernel_cnn2'), best_hps.get('kernel_cnn3'), best_hps.get('kernel_cnn4')],
      '\nunit:', [best_hps.get('units1'), best_hps.get('units2')],
      '\nkernel_regularizer:', [best_hps.get('kernel_regularizer1'), best_hps.get('kernel_regularizer2')],
      '\nrec_regularizer:', [best_hps.get('rec_regularizer1'), best_hps.get('rec_regularizer2')],
      '\nkernel_dense:', best_hps.get('kernel_dense'),
      '\nLearning rate:', best_hps.get('learning_rate'))
      
## Plot Model
from tensorflow.keras.utils import plot_model
model = tuner.hypermodel.build(best_hps)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

# Retrain model with the optimal hyperparameters
history = model.fit(x_train, y_train, batch_size=32, epochs = 10, validation_data = (x_test, y_test))

# Plot loss and accuracy graph
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0,5)
plt.show()

## Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

## Save Model
model.save('Data/model_CNN-BiLSTM_sentiment.h5')

## Upload new data
new_data = pd.read_csv('https://raw.githubusercontent.com/Syamsyuriani/Scrapping_Data/main/Quipper-Data.csv')

## Load Model
model = keras.models.load_model('Data/model_CNN-BiLSTM_sentiment.h5')

## Tokenization and pad sequencing
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(new_data['review'])
seq = tokenizer.texts_to_sequences(new_data['review'])
x_pred = keras.preprocessing.sequence.pad_sequences(seq, maxlen=80)
y_pred = model.predict(x_pred)

treshold = 0.5
for i in range(y_pred.shape[0]):
  if y_pred[i] > treshold:
    y_pred[i] = 1
  else:
    y_pred[i] = -1

new_data['label'] = y_pred

## The results of data that have been labeled
pd.set_option("max_colwidth", 100)
pd.set_option("max_rows", None)
new_data
