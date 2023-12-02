import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = None
encoder = None

def preprocess_data(data):
    
    pdata = []
    stopwords = nltk.corpus.stopwords.words("english")
    
    for i in data:
        i = i.replace('\s+',' ')
        i = i.replace('[-?:.!]', '')
        i = i.strip()
        i = i.lower()
        i = i.split()
        pdata.append(i)
    
    corpus =  list(map(lambda x: [item for item in x if item not in stopwords], pdata))
    pdata = []

    for s in corpus:
        pdata.append(s)

    for i in range(len(pdata)):
        pdata[i]=' '.join(pdata[i])
        
    return pdata

def initialize():
    
    global model
    global encoder
    nltk.download('stopwords')
    nltk.download('punkt')
    model = load_model('./Models/intent_model.h5')
    encoder = joblib.load('./Encoder/label_encoder.joblib')

def predict(query):

    vocab_size=1000
    max_tokens = 10
    
    preprocessed_query = preprocess_data([query])
    preprocessed_query = np.array(preprocessed_query)
    onehot_encoded = [hashing_trick(text=word, n=vocab_size, hash_function='md5') for word in preprocessed_query]
    padded_onehot_encoded =pad_sequences(onehot_encoded ,padding='pre',maxlen=max_tokens)
    result = model.predict(padded_onehot_encoded)
    result_class = np.argmax(result, axis=1)
    label = encoder.inverse_transform(result_class)[0]

    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        category = predict(query)
        return render_template('index.html', category=category)
    return render_template('index.html', category=None)

if __name__ == "__main__":
    initialize()
    app.run(host='0.0.0.0', debug=True, port=4000)
