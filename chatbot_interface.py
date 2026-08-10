from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from classifier import TopicClassifier
from rag import RetrievalAugmentedGeneration
from textwrap import dedent



class Chatbot:
    def __init__(self, classifier_path, rag_path, model_name: str= "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.classifier = TopicClassifier(classifier_path)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.chat_history_ids = None
        self.system_prompt = "<|system|>\nyou are a helpful assistant that classifies inputes and explains the results using additional context.<|end|>\n"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rag = RetrievalAugmentedGeneration(rag_path)

        print(f"Chatbot is using device: {self.device}")


    def encode_prompt(self, prompt: str):

        prompt = prompt + self.tokenizer.eos_token

        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    

        
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens = True)
        
    
    def generate_reply(self, prompt: str, prefix: str = "") -> str:
        prompt = prompt.strip()

        full_prompt = (
            self.system_prompt
            + "<|user|>\n"
            + prompt
            + "\n<|end|>\n"
            + "<|assistant|>\n"
            + prefix
        )

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][input_length:]
        continuation = self.decode_reply(new_tokens).strip()

        # Cleanup common bad generations.
        bad_starts = [
            "because ",
            "it ",
            "this is classified as ",
            "this is categorized as ",
            "this text is classed as ",
            "based on ",
            "\"",
        ]

        changed = True
        while changed:
            changed = False
            lower = continuation.lower().strip()
            for bad in bad_starts:
                if lower.startswith(bad):
                    continuation = continuation[len(bad):].strip()
                    changed = True

        reply = prefix + continuation

        # Remove quotes.
        reply = reply.replace('"', '').replace("'", "")

        # Keep only first line.
        reply = reply.split("\n")[0].strip()

        # Keep only one sentence.
        for end in [".", "!", "?"]:
                if end in reply:
                    reply = reply.split(end)[0].strip() + "."
                    break

                    
        if not reply.endswith("."):
                        reply += "."

        return reply.strip()
    
    def extract_text_to_classify(self, user_input):
        return user_input.strip()
    
    def classify_text(self, text):
        return self.classifier.classify_topic(text)
    
    def get_input(self):
        return input("You: ")
    
    def generate_friendly_reply(self, user_input, extracted_text, classification):





        context_texts = []
        retrieved_chunks = self.rag.retrieve(extracted_text)
        for chunk in retrieved_chunks:
            text = chunk[1]
            context_texts.append(text)
        context = "\n\n".join(context_texts)
        if not context:
            context = "No extra context was found"

        prompt = dedent(f"""
        You are writing a short explanation for a topic classifier.

        Text:
        {extracted_text}

        Predicted topic:
        {classification}

        Write only the reason after the word "because".
        Do not write the full sentence.
        Do not repeat the topic.
        Do not use quotation marks.
        Do not mention "text", "sentence", "instruction", "context", or "first sentence".
        Use only the user's words as evidence.
        Maximum 12 words.

        Good examples:
        mentions tariffs and international trade
        mentions China and production of chip and solar panel materials
        mentions airlines and lithium-ion batteries
        """).strip()

        prefix = f"This is classified as {classification} because "
        return self.generate_reply(prompt, prefix=prefix)

      




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
    bot = Chatbot("logistic_regression_topic_model.pkl",
                  "news_knowledge")
    bot.run()







