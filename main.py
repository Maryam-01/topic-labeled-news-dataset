from clean import read_news, explore_news, clean_news, save_clean_news


if __name__ == "__main__":
    news = read_news()

    if news is not None:
        print("\nBefore cleaning:")
        explore_news(news)

        cleaned_news = clean_news(news)

        print("\nAfter cleaning:")
        explore_news(cleaned_news)

        save_clean_news(cleaned_news)