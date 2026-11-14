"""Console based Student Feedback Analyzer (Python 3)."""

from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd


STOP_WORDS = {"and", "but", "the", "is", "a", "of", "too"}
REQUIRED_COLUMNS = {"StudentID", "Rating", "Comment"}


def load_feedback(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read and validate feedback rows; return valid records and warnings."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")

    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(sorted(missing)))

    valid_rows = []
    warnings = []
    for row_number, row in enumerate(data.to_dict(orient="records"), start=1):
        raw_rating = row["Rating"].strip()
        if not re.fullmatch(r"[1-5]", raw_rating):
            warnings.append(
                f"Warning: Row {row_number} skipped - rating must be a whole number from 1 to 5."
            )
            continue
        valid_rows.append(
            {"StudentID": row["StudentID"].strip(), "Rating": int(raw_rating), "Comment": row["Comment"].strip()}
        )

    return pd.DataFrame(valid_rows, columns=["StudentID", "Rating", "Comment"]), warnings


def sentiment_counts(feedback: pd.DataFrame) -> dict[str, int]:
    ratings = feedback["Rating"]
    return {
        "Positive": int(ratings.isin([4, 5]).sum()),
        "Neutral": int((ratings == 3).sum()),
        "Negative": int(ratings.isin([1, 2]).sum()),
    }


def top_words(feedback: pd.DataFrame, limit: int = 3) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for comment in feedback["Comment"]:
        words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", comment.lower())
        counts.update(word for word in words if word not in STOP_WORDS)
    # Sort by descending frequency, then alphabetically for ties.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def show_summary(feedback: pd.DataFrame) -> None:
    total = len(feedback)
    counts = feedback["Rating"].value_counts()
    average = feedback["Rating"].mean() if total else 0.0
    print(f"Valid feedback: {total}")
    print(f"Average rating: {average:.2f}")
    print("Ratings: " + ", ".join(f"{rating} -> {int(counts.get(rating, 0))}" for rating in (5, 4, 3, 2, 1)))
    words = top_words(feedback)
    formatted = ", ".join(f"{word} ({count})" for word, count in words) or "(no words)"
    print(f"Top 3 words: {formatted}")


def show_sentiment(feedback: pd.DataFrame) -> None:
    total = len(feedback)
    counts = sentiment_counts(feedback)
    for label, count in counts.items():
        percentage = count / total * 100 if total else 0.0
        print(f"{label}: {count} ({percentage:.2f}%)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart unavailable: install matplotlib with `pip install matplotlib`.")
        return

    labels = list(counts)
    values = [counts[label] for label in labels]
    plt.bar(labels, values, color=["#2e8b57", "#e0ad35", "#c94c4c"])
    plt.title("Student Feedback Sentiment")
    plt.ylabel("Number of responses")
    plt.tight_layout()
    plt.show()


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("feedback.csv")
    try:
        feedback, warnings = load_feedback(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        print(f"Error: {error}")
        return

    for warning in warnings:
        print(warning)
    print(f"Loaded {len(feedback)} valid feedback response(s) from {csv_path.name}.")

    while True:
        print("\nStudent Feedback Analyzer")
        print("1. Summary and top 3 words")
        print("2. Sentiment percentages and bar chart")
        print("3. Exit")
        choice = input("Choose an option (1-3): ").strip()
        if choice == "1":
            show_summary(feedback)
        elif choice == "2":
            show_sentiment(feedback)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
