"""Streamlit interface for the SMS spam classifier."""

import streamlit as st

from model import predict_message, train_model


@st.cache_resource
def get_model():
    """Train the model once and reuse it across Streamlit reruns."""
    return train_model()


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")
st.title("SMS Spam Classifier")
st.write("Enter an SMS message to classify it as **ham** or **spam**.")

message = st.text_area("SMS message", placeholder="Type or paste a message here...")

if st.button("Classify message", type="primary"):
    if not message.strip():
        st.warning("Please enter an SMS message first.")
    else:
        label, confidence = predict_message(get_model(), message)
        if label == "spam":
            st.error(f"Spam ({confidence:.1%} confidence)")
        else:
            st.success(f"Ham ({confidence:.1%} confidence)")
