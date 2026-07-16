import json
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "complaints.json")


def load_complaints():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_complaints(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_duplicate(new_complaint, category):
    stored_complaints = load_complaints()
    category_complaints = stored_complaints.get(category, [])

    if not category_complaints:
        category_complaints.append(new_complaint)
        stored_complaints[category] = category_complaints
        save_complaints(stored_complaints)

        return {
            "is_duplicate": False,
            "matched_complaint": None,
            "similarity": 0.0
        }

    stored_embeddings = model.encode(category_complaints)
    new_embedding = model.encode([new_complaint])

    similarities = cosine_similarity(new_embedding, stored_embeddings)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score > 0.50:
        return {
            "is_duplicate": True,
            "matched_complaint": category_complaints[best_index],
            "similarity": round(float(best_score), 3)
        }

    category_complaints.append(new_complaint)
    stored_complaints[category] = category_complaints
    save_complaints(stored_complaints)

    return {
        "is_duplicate": False,
        "matched_complaint": None,
        "similarity": round(float(best_score), 3)
    }