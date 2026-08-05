import pandas as pd
import matplotlib.pyplot as plt
from clean import read_news, clean_news
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_models():
    # 1. Read and clean data
    news = read_news()
    cleaned = clean_news(news)
    print(cleaned["topic"].value_counts())

    # 2. Select features and target
    X = cleaned["title"]
    y = cleaned["topic"]

    # 3. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Naive Bayes model
    nb_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", MultinomialNB())
    ])

    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)

    print("\nNaive Bayes Results")
    print("-------------------")
    print("Accuracy:", accuracy_score(y_test, nb_predictions))
    print(classification_report(y_test, nb_predictions))

    # 5. Logistic Regression model
    lr_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    print("\nLogistic Regression Results")
    print("---------------------------")
    print("Accuracy:", accuracy_score(y_test, lr_predictions))
    print(classification_report(y_test, lr_predictions))
    with open("logistic_regression_topic_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    print("Model saved successfully.")

    


    ConfusionMatrixDisplay.from_predictions(
        y_test,
        lr_predictions,
        xticks_rotation=45,
        cmap="Blues",
        normalize="true"
    )

    plt.title("Normalized Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.show()

    return nb_model, lr_model


if __name__ == "__main__":
    train_models()