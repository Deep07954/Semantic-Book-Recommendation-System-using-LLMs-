import threading
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import threading

load_dotenv()

df = pd.read_csv("books_with_emotions.csv")
df["authors"] = df["authors"].fillna("Unknown Author")

df["large_thumbnail"] = df["thumbnail"] + "&fife=w800"
df["large_thumbnail"] = np.where(
    df["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    df["large_thumbnail"],
)

raw_documents = TextLoader("tagged_descriptions.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = df[df["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_category"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(df["simple_category"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)



def launch_app(share=False):
    threading.Thread(
        target=lambda: dashboard.launch(
            server_name="0.0.0.0",
            server_port=None,
            share=share,
            inbrowser=True
        ),
        daemon=True
    ).start()

def start_dashboard(enable_share):
    launch_app(share=enable_share)
    return f"✅ Dashboard launched with {'public share link' if enable_share else 'local access only'}"

# --- Control panel to launch dashboard ---
with gr.Blocks(theme=gr.themes.Soft(), title="Dashboard Launcher") as control_panel:
    gr.Markdown("### 🚀 Launch Your Semantic Book Recommender Dashboard")
    share_toggle = gr.Checkbox(label="Enable Public Share Link?", value=False)
    launch_button = gr.Button("Launch Dashboard")
    output_text = gr.Textbox(label="Status", interactive=False)
    launch_button.click(fn=start_dashboard, inputs=share_toggle, outputs=output_text)

# --- Run control panel ---
if __name__ == "__main__":
    control_panel.launch(inbrowser=True)
