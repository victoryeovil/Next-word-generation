import numpy as np
import pickle
import streamlit as st
from pickle import load
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical



tokenizer = Tokenizer()
# loading the saved model
model = load_model('nextword1.h5')
# load the tokenizer
tokenizer = load(open( 'token.pkl' , 'rb' ))


def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = np.argmax(model.predict(encoded))

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)


def main():
    
    
    # giving a title
    st.title('NEXT WORD PREDICTOR')
    st.text('R207764L Victor Marisa')
    
    
    # getting the input data from the userocco
    
    
    sentance = st.text_input('Enter a five word sentence')

    
    
    # code for Prediction
    word = ''
    
    # creating a button for Prediction
    
    if st.button('view predicted word'):
      if len(sentance.split()) <5 :
        st.success('please enter a five word sentence')
      else:
        word = generate_text_seq(model, tokenizer, 5, sentance, 1)
        
        
    st.success(word)
    
    
    
    
    
if __name__ == '__main__':
    main()
