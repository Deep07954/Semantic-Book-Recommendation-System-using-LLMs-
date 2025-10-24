import streamlit as st # type: ignore
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings # pyright: ignore[reportMissingImports]
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Load Data
df = pd.read_csv("books_with_emotions.csv")
df["authors"] = df["authors"].fillna("Unknown Author")

df["large_thumbnail"] = df["thumbnail"] + "&fife=w800"
df["large_thumbnail"] = np.where(
    df["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    df["large_thumbnail"],
)

# Load Documents DB with caching
@st.cache_resource()
def load_database():
    raw_documents = TextLoader("tagged_descriptions.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(
        documents,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    return db

db_books = load_database()


# Recommendation Logic
def retrieve_semantic_recommendations(query, category, emotion_weights, top_k=32):

    recs = db_books.similarity_search(query, k=200)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = df[df["isbn13"].isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_category"] == category]

    for emotion, weight in emotion_weights.items():
        book_recs[f"score_{emotion}"] = book_recs[emotion] * weight

    book_recs["total_score"] = book_recs[[f"score_{e}" for e in emotion_weights]].sum(axis=1)
    book_recs = book_recs.sort_values(by="total_score", ascending=False).head(top_k)

    return book_recs


# UI
st.title("📚 Semantic Book Recommender")
st.write("Discover books that match your story, mood, and soul.")

query = st.text_input("Describe a book, theme, or feeling", placeholder="A heartwarming story about friendship")

# Sidebar Controls
st.sidebar.title("🎛️ Filters")

category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["simple_category"].unique())
)

st.sidebar.subheader("🎭 Emotional Tone Weight")

emotion_weights = {
    "joy": st.sidebar.slider("😊 Joy", 0.0, 1.0, 0.3),
    "surprise": st.sidebar.slider("😲 Surprise", 0.0, 1.0, 0.1),
    "anger": st.sidebar.slider("😡 Anger", 0.0, 1.0, 0.0),
    "fear": st.sidebar.slider("😱 Suspense", 0.0, 1.0, 0.2),
    "sadness": st.sidebar.slider("😢 Sadness", 0.0, 1.0, 0.1),
}

st.sidebar.markdown("---")
top_k = st.sidebar.slider("How many books to show?", 8, 32, 16)

if st.button("🔍 Find Books"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Finding the right books just for you..."):
            recommendations = retrieve_semantic_recommendations(
                query, category, emotion_weights, top_k
            )

        if recommendations.empty:
            st.error("No matching books found.")
        else:
            st.subheader("✨ Your Recommendations")
            for _, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(row["large_thumbnail"], width=110)

                with col2:
                    authors = row["authors"].replace(";", ", ")
                    st.markdown(f"### {row['title']}")
                    st.caption(f"by {authors}")

                    desc = " ".join(row["description"].split()[:40]) + "..."
                    st.write(desc)

                st.markdown("---")

else:
    st.info("Adjust filters in the sidebar and click the button!")
