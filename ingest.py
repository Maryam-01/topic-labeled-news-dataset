import requests
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def lambda_handler(event=None, context=None):
    url = (
        "https://raw.githubusercontent.com/kotartemiy/topic-labeled-news-dataset/master/labeled_newscatcher_dataset.csv"

    )



    output_file = Path("data/raw/labeled_newscatcher_dataset.csv")
    
    


    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        output_file.write_bytes(response.content)
        logger.info(f"saved dataset to {output_file}")

        return {
        "statusCode": 200,
        "message":"dataset downloaded successfully",
        "file_path": str(output_file),
        "bytes_downloaded": len(response.content),

    }
    except Exception as e:
        logger.exception("failed to download dataset")
        return {
             "statusCode": 500,
             "message": "failed to download dataset",
             "error": str(e),
        }
    
if __name__ == "__main__":
    print(lambda_handler())

        