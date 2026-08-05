from classifier import TopicClassifier


class SimpleInterface:
    def __init__(self, filepath):
        self.filepath = filepath
        self.classifier = TopicClassifier(filepath)

    def get_input(self):
        insert_headline = input("insert headline: ").strip()
        return insert_headline
    
    def classify_headline(self, insert_headline):
        result = self.classifier.classify_topic(insert_headline)
        return result
    
    def display_result(self, result):
        
        print(f"[Result] {result}")

        
    def run(self):
        print("News Topic Classifier")
        print("Type 'quit' to stop.\n")
        while True:
            headline = self.get_input()
            if headline.lower() == "quit":
                break
            if not headline:
                continue
            
            result = self.classify_headline(headline)
            self.display_result(result)
            print()

if __name__ == "__main__":
    interface = SimpleInterface("logistic_regression_topic_model.pkl")
    interface.run()
            
        



