from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from classifier import TopicClassifier



class Chatbot():
    def __init__(self, classifier_path, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.classifier = TopicClassifier(classifier_path)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.chat_history_ids = None
        self.system_prompt = "<|system|>\nyou are a helpful assistant that explains classification results clearly.<|end|>\n"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Chatbot is using device: {self.device}")


    def encode_prompt(self, prompt: str):

        prompt = prompt + self.tokenizer.eos_token

        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens = True)
        
    
    def generate_reply(self, prompt: str) -> str:

        prompt = prompt.strip()

        full_prompt = (
            self.system_prompt
            + "<|user|>\n"
            + prompt
            + "\n<|end|>\n"
            + "<|assistant|>\n"
        )
        


        inputs = self.tokenizer(
        full_prompt,
        return_tensors="pt"
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=self.tokenizer.eos_token_id
    )

        new_tokens = output_ids[0][input_length:]
        reply = self.decode_reply(new_tokens)

        

        return reply.strip()
    
    def extract_text_to_classify(self, user_input):
        return user_input.strip()
    
    def classify_text(self, text):
        return self.classifier.classify_topic(text)
    
    def get_input(self):
        return input("You: ")
    
    def generate_friendly_reply(self, user_input, extracted_text, classification):
        
        return f'I classified "{extracted_text}" as {classification}.'


    def run(self):
        while True:
            user_input = self.get_input()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            extracted_text = self.extract_text_to_classify(user_input)
            classification = self.classify_text(extracted_text)

            reply = self.generate_friendly_reply(
                user_input,
                extracted_text,
                classification
            )

            print("Bot:", reply)

    


    

if __name__ == "__main__":
    bot = Chatbot("logistic_regression_topic_model.pkl")
    bot.run()







