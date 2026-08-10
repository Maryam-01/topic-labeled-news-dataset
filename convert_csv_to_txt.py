import pandas as pd
from pathlib import Path


csv_path = "bbc_news.csv"
output_dir = Path("news_knowledge")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

articles_per_category = 100

for category, group in df.groupby("Category"):
    selected_articles = group.sample(
        n=min(len(group), articles_per_category),
        random_state=42
    )
    file_path = output_dir / f"{category}_articles.txt"


    with open(file_path, "w", encoding="utf-8") as f:
        for _, row in selected_articles.iterrows():
            article_id = row["ArticleId"]
            text = row["Text"]

            f.write(f"Article ID: {article_id}\n")
            f.write(f"Category: {category}\n")
            f.write("Text:\n")
            f.write(text)
            f.write("\n\n---END OF ARTICLE ----\n\n")

print("done creating one txt file per category")


 

csv_path = "bbc_news.csv"
output_dir = Path("knowledge_news")
output_dir.mkdir(exist_ok=True)


articles_per_category = 100

for category, group in df.groupby("category"):
    selected_articles = group.sample(
        n=min(len(group), articles_per_category, random_state=42)
    )

    