import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):

    text = text.lower()


    text = nltk.word_tokenize(text)


    text = [i for i in text if i.isalnum()]


    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]


    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]


    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("message/sms spam Classifier")
input_sms = st.text_input("Enter the message")

#1. preprocess
transformed_sms = transform_text(input_sms)

vector_input = tfidf.transform([transformed_sms])

result = model.predict(vector_input)[0]

if result == 1
    st.header("spam")
else:
    st.header("Not spam")


