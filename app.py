import streamlit as st
from translate import translate

st.title("English to Romanian Translator")

st.subheader("Enter English Text to Translate")
english_text = st.text_area(label='', height=150)

epoch_choice = '09'

if st.button("Translate to Romanian"):
    if epoch_choice is None:
        st.error("Please select a model epoch first.")
    elif not english_text:
        st.warning("Please enter English text to translate.")
    else:
        with st.spinner(f"Translating with Epoch {epoch_choice}..."):
            try:
                romanian_translation = translate(english_text, epoch_choice)
                st.subheader("Romanian Translation:")
                st.write(romanian_translation)
            except Exception as e:
                st.error(f"Translation Error: {e}")
                st.error("Please check your model files and code. "
                        "Make sure the 'translate' function and model loading are working correctly.")