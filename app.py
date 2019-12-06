from flask import Flask,jsonify,Response
from flask import render_template
from flask import request
from keras.models import load_model
from keras import backend as K
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
en_stop = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__) 

#model dependencies
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def hamming_loss(y_true, y_pred):
  return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

#loadin
dependencies = {'f1_m': f1_m, 'recall_m':recall_m, 'precision_m':precision_m, 'hamming_loss':hamming_loss}
movie_lstm_single_10 = load_model('movie_lstm_single_10.h5', custom_objects= dependencies)
with open('tokenizer_movie.pickle', 'rb') as handle:
    tokenizer_movie = pickle.load(handle)


@app.route('/get_predictions',methods = ['GET','POST'])
def get_classification_labels():
	# #get the request data
	if request is None:
		return Response('No request',status = 400)

	if not request.args:
		return Response('No json provided.', status = 400)
	if 'plot' not in request.args:
		return Response('No plot provided.', status = 400)
	else:

		plot = request.args['plot']
		labels = predict_genre(plot,tokenizer_movie,movie_lstm_single_10)
		if len(labels) == 0:
			genre = 'Please enter valid plot summary..'
		else:
			genre = labels
		return jsonify({"status": "success", "labels": genre})



def predict_genre(new_data,tokenizer,model_file):
  CATEGORIES = ["Drama", "World cinema", "Action", "Black-and-white", "Romance Film", "Thriller", "Comedy", "Short Film"	]
  new_data = preprocess_text(new_data)
  X_tokenized = tokenizer.texts_to_sequences(new_data)
  to_pad = []
  for x in X_tokenized:
      if len(x) >0:
          to_pad.append(x[0])
          
  while len(to_pad) < 500:
      to_pad.append(0)
      
  test = [x for x in to_pad]
  test = [test]
  test = np.array(test)
  
  predictions = model_file.predict(test)
  final_labels = []
  for i in predictions:
    for j in range(len(i)):
      if i[j]> 0.5:
        final_labels.append(CATEGORIES[j])
  return final_labels



# Preprocess input text 
def preprocess_text(document):
    #now = datetime.datetime.now()
    
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    
    tokens = document.split()
    
    #### Remove stopwords
    words = [w for w in tokens if w not in stopwords.words('english')]
    words = [word for word in words if word not in en_stop]
    
    #### Lemmatize tokens obtained after removing stopwords
    wnl = WordNetLemmatizer()
    tagged = nltk.pos_tag(words)
    lem_list = []
    for word, tag in tagged:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None 
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)
        lem_list.append(lemma)
        
    #preprocessed_text = ' '.join(lem_list)
    
    return lem_list

if __name__ == "__main__":
	app.run()

