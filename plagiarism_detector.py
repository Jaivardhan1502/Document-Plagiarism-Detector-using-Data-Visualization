import os
import itertools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

FOLDER = "documents"  # Folder containing .txt files

# ------------------- Load Documents -------------------
def load_documents(folder):
    docs = {}
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            with open(os.path.join(folder, f), "r", encoding="utf-8") as file:
                docs[f] = file.read()
    return docs

# ------------------- Compute Similarity -------------------
def compute_similarity(docs):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(docs.values())
    sim_matrix = cosine_similarity(tfidf)
    return sim_matrix

# ------------------- Show Results -------------------
def show_results(docs, sim_matrix, threshold=0.3):
    names = list(docs.keys())
    print("\n📄 Total documents loaded:", len(names))
    print("Comparing files...\n")

    # --- Text similarity summary ---
    for i, j in itertools.combinations(range(len(names)), 2):
        sim = sim_matrix[i][j]
        print(f"{names[i]} ↔ {names[j]} : Similarity = {sim*100:.1f}%")
        if sim >= threshold:
            print(f"⚠️ Possible plagiarism between '{names[i]}' and '{names[j]}'\n")

    # --- Heatmap ---
    plt.figure(figsize=(6,5))
    sns.heatmap(sim_matrix, annot=True, xticklabels=names, yticklabels=names, cmap="YlGnBu")
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.show()

    # --- Word Cloud ---
    all_text = " ".join(docs.values())
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Common Words Word Cloud")
    plt.tight_layout()
    plt.show()

# ------------------- Main -------------------
if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        print(f"❌ Folder '{FOLDER}' not found. Please create it.")
    else:
        docs = load_documents(FOLDER)
        if len(docs) < 2:
            print("⚠️ Need at least 2 text files in 'documents' to compare.")
        else:
            sim = compute_similarity(docs)
            show_results(docs, sim, threshold=0.4)
