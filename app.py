import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config (important)
# -----------------------------
st.set_page_config(page_title="SmartLaw", page_icon="⚖️")

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("ipc_fir_cleaned.csv")

# -----------------------------
# Train TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["text"])

# -----------------------------
# Chatbot logic
# -----------------------------
def get_legal_response(query):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]
    return best_index, best_score

# -----------------------------
# UI Header
# -----------------------------
st.title("⚖️ SmartLaw: From Confusion to Conclusion")
st.caption("An NLP-Based Legal Awareness Chatbot")

# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input (THIS IS KEY)
# -----------------------------
user_input = st.chat_input("Ask about any IPC section or legal situation...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    index, confidence = get_legal_response(user_input)

    if confidence < 0.15:
        bot_reply = "❌ No relevant IPC section found for your query."
    else:
        row = data.iloc[index]
        bot_reply = (
            f"**IPC Section {row['section']}**\n\n"
            f"**Description:** {row['description']}\n\n"
            f"**Punishment:** {row['punishment']}\n\n"
            f"_Confidence score: {round(confidence, 2)}_"
        )

    # Save bot message
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_reply)


