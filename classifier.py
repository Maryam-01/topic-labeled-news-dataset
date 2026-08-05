import pickle

class TopicClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        
        with open(self.filepath, "rb") as f:
            self.model = pickle.load(f)

    def preprocess(self, headline):
        headline = headline.strip().lower()
        return headline
    
    def classify_topic(self, headline):
        cleaned_headline = self.preprocess(headline)
        prediction = self.model.predict([cleaned_headline])
        print("Predicted topic:", prediction[0])

        return prediction[0]

