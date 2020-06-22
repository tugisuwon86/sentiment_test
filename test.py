import streamlit as st
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

def load_model():
    
    # load model
    filename = 'linearsvc_sentiment_model_light.sav'
    loaded_model = joblib.load(filename)

    # tfidf model load test
    tfidf_vect_load = pickle.load(open("tfidf_yelp_feature_light.pkl", "rb"))
    return loaded_model, tfidf_vect_load

# initiate
loaded_model, tfidf_vect_load = load_model()

st.write("""
# Simple text sentiment analysis test tool
This app predicts **sentiment score** of a document!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    text_sentiment = st.text_input("label goes here", "na")
    data = {'text_sentiment': text_sentiment}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

current = tfidf_vect_load.transform(df["text_sentiment"].values)

prediction = loaded_model.predict(current)
#prediction_proba = loaded_model.predict_proba(current)

st.subheader('Prediction using SVC')
st.write(["negative", "positive"][prediction[0]])


# vader
analyzer = SentimentIntensityAnalyzer()
temp = analyzer.polarity_scores(df["text_sentiment"].values[0])
st.subheader('Prediction using Vader')
st.write(temp["compound"])