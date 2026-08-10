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
        
        probabilities = self.model.predict_proba([cleaned_headline])[0]
        topic_probabilities = list(zip(self.model.classes_, probabilities))
        topic_probabilities = sorted(
            topic_probabilities,
            key=lambda item: item[1],
            reverse = True
        )
        for topic, probability in topic_probabilities:
            percent = round(probability*100)
            print(topic,str(percent) + "%")

        return prediction[0]
    
topic = TopicClassifier("logistic_regression_topic_model.pkl")
topic.classify_topic("new study shows diabetes medicine on market")

