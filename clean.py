import pandas as pd
from pathlib import Path

def read_news():
    try:
        news = pd.read_csv("data/raw/labeled_newscatcher_dataset.csv", sep= ";")
        if news.empty:
            print("file is empty")
        else:
            return news
    except FileNotFoundError:
        print("csv file not found")




def explore_news(news):
    print("\n---BASIC INFO---")
    print("shape:", news.shape)
    print(news.info())

    print("\n---FIRST ROWS---")
    print(news.head())

    print("\n--MISSING VALUES---")
    print(news.isna().sum())

    print("\n---TOPIC COUNTS---")
    print(news["topic"].value_counts())

    print("\n---LANGUAGE COUNTS---")
    print(news["lang"].value_counts())

    print("\n---DATE RANGE---")
    print("Earliest:", news["published_date"].min())
    print("Latest", news["published_date"].max())

    print("\n--- TOP DOMAINS ---")
    print(news["domain"].value_counts().head(20))

    print("\n--- DUPLICATES ---")
    print("Duplicate rows:", news.duplicated().sum())
    print("Duplicate links:", news["link"].duplicated().sum())
    print("Duplicate titles:", news["title"].duplicated().sum())



CLEAN_DATA_PATH = Path("data/processed/cleaned_news.csv")
def clean_news(news):
    print("original shape:", news.shape)

    news = news.drop_duplicates()
    print("after dropping duplicates rows:", news.shape)

    news= news.drop_duplicates(subset=["link"])
    print("after dropping duplicate links:", news.shape)

    news = news.drop_duplicates(subset=["title"])
    print("after dropping duplicate titles:", news.shape)

    news["title"] = news["title"].str.strip()
    news["topic"] = news["topic"].str.strip().str.lower()
    news["domain"] = news["domain"].str.strip().str.lower()
    news["lang"] = news["lang"].str.strip().str.lower()

    news= news[news["lang"] == "en"]
    print("after keeping english rows:", news.shape)
    news = news.dropna(subset=["title", "link", "topic", "lang"])

    news = news[news["title"] != ""]
    print("after removing empty titles:", news.shape)
    
    news = news[news["topic"] != ""]
    print("after removing empty topic:", news.shape)

    news = news[news["domain"] != ""]
    print("after removing empty domain:", news.shape)

    news = news[news["lang"] != ""]
    print("after removing empty lang:", news.shape)

    news = news[news["published_date"] != ""]
    print("after removing empty published_date:", news.shape)

    news = news[news["link"] != ""]
    print("after removing empty link:", news.shape)

    news["published_date"] = pd.to_datetime(
        news["published_date"],
        errors="coerce"
    )
    return news

def save_clean_news(news):
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(CLEAN_DATA_PATH, index= False)
    print(f"saved cleaned dataset to: {CLEAN_DATA_PATH}")
    print("final shape:", news.shape)

if __name__ == "__main__":
    news = read_news()

    if news is not None:
        print("\nBefore cleaning:")
        explore_news(news)

        cleaned_news = clean_news(news)

        print("\nAfter cleaning:")
        explore_news(cleaned_news)

        save_clean_news(cleaned_news)







