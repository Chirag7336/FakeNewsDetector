import streamlit as st
import joblib

Vectoization = joblib.load("Vectorization.jb")
model = joblib.load("LR_model.jb")

st.title("Fake_News_Detector")
st.write("Enter a News Article to check it is Real or Fake")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = Vectoization.transform([news_input])
        prediction = model.predict(transform_input)
        
        if prediction[0]==1:
            st.success("The news is Real")
        else:
            st.error("The news is Fake")
            
    else:
        st.warning("please some other text to analysis")