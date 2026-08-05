import pickle

class TopicClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        
        with open(self.filepath, "rb") as f:
            self.model = pickle.load(f)


    def classify_topic(self):
        insert_headline = input("Enter a headline: ")
        prediction = self.model.predict([insert_headline])
        print("Predicted topic:", prediction[0])

        return prediction[0]

classifier = TopicClassifier("logistic_regression_topic_model.pkl")
classifier.classify_topic()