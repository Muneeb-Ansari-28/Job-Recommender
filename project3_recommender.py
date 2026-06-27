"""
Project 3: AI Recommendation Logic - Tech Stack Recommender
DecodeLabs Industrial Training Kit - AI Track

Goal: Map a user's raw skills/interests to job roles using Content-Based
Filtering: TF-IDF vectorization + Cosine Similarity.

Pipeline (IPO + 4-step ranking):
    Ingestion -> Scoring (Cosine Similarity) -> Sorting -> Filtering (Top-N)

Requirements:
    pip install scikit-learn pandas --break-system-packages
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "raw_skills.csv")


def load_dataset(path=CSV_PATH) -> pd.DataFrame:
    """
    STEP 1: Ingestion - load the job roles dataset from CSV file.
    The CSV file must exist in the same directory as this script.
    """
    if not os.path.exists(path):
        print(f"❌ ERROR: '{path}' not found!")
        print(f"   Expected location: {path}")
        print(f"\n   Please create the CSV file with the following format:")
        print(f"   role,skills")
        print(f"   Data Scientist,Python SQL Machine Learning Data Analysis...")
        print(f"\n   Or copy the raw_skills.csv file to the Project3 directory.")
        exit(1)
    
    # Load and return data from CSV file
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} job roles from: {path}\n")
    return df


def get_user_input() -> list:
    """STEP 1: Ingestion - capture user state (minimum 3 skills)."""
    print("Enter at least 3 skills/interests, separated by commas.")
    print("Example: Python, Cloud Computing, Automation")
    raw = input("\nYour skills: ")
    skills = [s.strip() for s in raw.split(",") if s.strip()]

    while len(skills) < 3:
        print("Please enter at least 3 skills.")
        raw = input("Your skills: ")
        skills = [s.strip() for s in raw.split(",") if s.strip()]

    return skills


def build_vector_space(df: pd.DataFrame, user_skills: list):
    """
    PROCESS: Vector Mapping with TF-IDF.
    User profile and item features are mapped to the SAME vocabulary space.
    """
    user_profile_text = " ".join(user_skills)

    # Combine items + user profile so TF-IDF builds one shared vocabulary
    corpus = df["skills"].tolist() + [user_profile_text]

    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    item_vectors = tfidf_matrix[:-1]   # all rows except last
    user_vector = tfidf_matrix[-1]     # last row = user profile

    return item_vectors, user_vector


def score_and_rank(df: pd.DataFrame, item_vectors, user_vector, top_n=3):
    """
    STEP 2: Scoring - Cosine Similarity between user vector and each item.
    STEP 3: Sorting - descending order by score.
    STEP 4: Filtering - Top-N results.
    """
    scores = cosine_similarity(user_vector, item_vectors).flatten()

    results = df.copy()
    results["match_score"] = scores
    results = results.sort_values(by="match_score", ascending=False)

    return results.head(top_n)


def main():
    print("=" * 55)
    print(" DecodeLabs Tech Stack Recommender (Content-Based) ")
    print("=" * 55)

    df = load_dataset()
    user_skills = get_user_input()

    item_vectors, user_vector = build_vector_space(df, user_skills)
    top_matches = score_and_rank(df, item_vectors, user_vector, top_n=3)

    print("\nTop 3 Recommended Career Paths:\n")
    for rank, (_, row) in enumerate(top_matches.iterrows(), start=1):
        print(f"{rank}. {row['role']}  (Match Score: {row['match_score']:.2f})")
        print(f"   Key Skills: {row['skills']}\n")


if __name__ == "__main__":
    main()
