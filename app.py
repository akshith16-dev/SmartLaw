import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Load the cleaned IPC dataset
# ------------------------------------
data = pd.read_csv("ipc_fir_final_cleaned.csv")

# ------------------------------------
# Train TF-IDF Model
# ------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["text"])

# ------------------------------------
# Chatbot Logic with Confidence Score
# ------------------------------------
def get_legal_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]
    
    return best_index, best_score

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.title("⚖️ SmartLaw: From Confusion to Conclusion")
st.caption("An NLP-Based Legal Awareness Chatbot")

st.markdown("""
**How it works:**
- Enter a legal situation or IPC section  
- The system finds the most relevant law  
- Results are for legal awareness only  
""")

st.markdown("""
**Example queries:**
- Police threatened me with cheating case  
- Wearing police uniform illegally  
- Helping prisoner escape  
- Harbouring criminal  
""")

# User input
user_input = st.text_input("Enter your legal query:")

# ------------------------------------
# Display Result
# ------------------------------------
if user_input:
    index, confidence = get_legal_response(user_input)

    # Confidence threshold
    if confidence < 0.15:
        st.warning("No relevant IPC section found for your query.")
    else:
        result = data.iloc[index]
        st.subheader(f"IPC Section {result['section']}")
        st.write("**Description:**", result["description"])
        st.write("**Punishment:**", result["punishment"])
        st.caption(f"Similarity confidence score: {round(confidence, 2)}")

# ------------------------------------
# Sidebar Information
# ------------------------------------
st.sidebar.title("About SmartLaw")
st.sidebar.write(
    "SmartLaw is an NLP-based legal awareness chatbot "
    "that uses TF-IDF and cosine similarity to help users "
    "understand Indian legal sections in a simple manner."
)