import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from text_utils import augment_complaint_text, normalize_complaint_text

DATASET_PATH = os.path.join(os.path.dirname(__file__), "complaints_dataset.csv")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        category = row["category"]
        priority = row["priority"]
        for variant in augment_complaint_text(text):
            rows.append(
                {
                    "text": variant,
                    "category": category,
                    "priority": priority,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df["text"] = df["text"].astype(str).map(normalize_complaint_text)

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    train_augmented = build_training_frame(train_df)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_augmented["text"])
    X_test = vectorizer.transform(test_df["text"])

    category_model = LinearSVC(class_weight="balanced", C=0.5)
    category_model.fit(X_train, train_augmented["category"])

    priority_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )
    priority_model.fit(X_train, train_augmented["priority"])

    category_pred = category_model.predict(X_test)
    priority_pred = priority_model.predict(X_test)

    print("Category accuracy:", round(accuracy_score(test_df["category"], category_pred), 3))
    print(classification_report(test_df["category"], category_pred, zero_division=0))
    print("Priority accuracy:", round(accuracy_score(test_df["priority"], priority_pred), 3))
    print(classification_report(test_df["priority"], priority_pred, zero_division=0))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    with open(os.path.join(WEIGHTS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(WEIGHTS_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(category_model, f)

    with open(os.path.join(WEIGHTS_DIR, "priority_classifier.pkl"), "wb") as f:
        pickle.dump(priority_model, f)

    print("Training complete. Updated models saved to weights/")


if __name__ == "__main__":
    main()
