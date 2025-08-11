import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# with open('transform_text.pkl', 'rb') as file:
#     loaded_function = pickle.load(file)
vectorizer = pickle.load(open('vectorizer.pkl' , 'rb'))
model = pickle.load(open('model.pkl' , 'rb'))


st.title("Email/SMS Spam Classification")
message = st.text_area(label=" " , placeholder="Enter the SMS/Email here")

stop_words = stopwords.words('english')# print(stop_words)
punctuations = string.punctuation
porter = PorterStemmer()

def transform_text(text):
        text = text.lower() # to lower case

        y = ""
        for word in text.split():
            if word not in stop_words and word not in punctuations: # removing the stop words and punctuation marks .
                y += word+" "
                
        text = y
        y = ""
        for word in text.split():
            y += porter.stem(word)+" " # replacing words with there stem words .
        text = y
        return text.rstrip()


if st.button(label='Predict'):
     transformed_text = transform_text(message)
     print(transformed_text)
     vectorized_message = vectorizer.transform([transformed_text])
     print(vectorized_message)
     result = model.predict(vectorized_message)[0]
     print(result)
     if result == 0:
          st.header(body="NOT SPAM")
     elif result == 1:
          st.header(body="SPAM")
    


    
    


