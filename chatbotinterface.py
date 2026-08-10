from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from classifier import TopicClassifier
from rag import RetrievalAugmentedGeneration
from textwrap import dedent
import re


class Chatbot:
    def __init__(
        self,
        classifier_path,
        rag_path,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ):
        self.classifier = TopicClassifier(classifier_path)
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.rag = RetrievalAugmentedGeneration(rag_path)

        print(f"Chatbot is using device: {self.device}")

    def decode_reply(self, reply_ids) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)

    def generate_reply(self, prompt: str) -> str:
        prompt = prompt.strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classification explanation assistant. "
                    "Follow the user's output rules exactly. "
                    "Do not continue articles or retrieved context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            full_prompt = (
                "System: You are a strict classification explanation assistant.\n"
                f"User: {prompt}\n"
                "Assistant:"
            )

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
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

    def parse_classification(self, classification):
        """
        Accepts either:
        1. A dict like:
           {"technology": 100, "sports": 0, ...}

        2. A string like:
           Predicted topic: technology
           technology 100%
           entertainment 0%
           ...

        Returns:
        topic, score, topics
        """

        topics = {}

        if isinstance(classification, dict):
            for key, value in classification.items():
                try:
                    topics[str(key).lower()] = float(value)
                except Exception:
                    continue

            if not topics:
                return None, 0.0, {}

            topic = max(topics, key=topics.get)
            score = topics[topic]
            return topic, score, topics

        text = str(classification)

        predicted_match = re.search(
            r"Predicted topic:\s*([a-zA-Z_ -]+)",
            text,
            re.IGNORECASE
        )

        predicted_topic = None

        if predicted_match:
            predicted_topic = predicted_match.group(1).strip().lower()

        for line in text.splitlines():
            line = line.strip()

            match = re.match(r"^([a-zA-Z_ -]+)\s+([0-9]+(?:\.[0-9]+)?)%$", line)

            if match:
                label = match.group(1).strip().lower()
                score = float(match.group(2))
                topics[label] = score

        if predicted_topic:
            score = topics.get(predicted_topic, 0.0)
            return predicted_topic, score, topics

        if topics:
            topic = max(topics, key=topics.get)
            score = topics[topic]
            return topic, score, topics

        return None, 0.0, {}

    def generate_friendly_reply(self, user_input, extracted_text, classification):
        topic, score, topics = self.parse_classification(classification)

        if topic is None:
            return "I could not identify a clear topic from the text."

        context = "None"
        use_rag = len(extracted_text.split()) >= 8

        if use_rag:
            try:
                retrieved_chunks = self.rag.retrieve(extracted_text)
                context_texts = []

                for chunk in retrieved_chunks[:1]:
                    if isinstance(chunk, tuple):
                        text = str(chunk[1])
                    else:
                        text = str(chunk)

                    context_texts.append(text[:300])

                if context_texts:
                    context = "\n".join(context_texts)

            except Exception:
                context = "None"

        prompt = dedent(f"""
        TASK:
        Write one short explanation for a news topic classifier.

        TEXT:
        "{extracted_text}"

        PREDICTED_TOPIC:
        {topic}

        CONFIDENCE:
        {score:.0f}%

        CONTEXT:
        {context}

        OUTPUT RULES:
        - Output exactly one sentence.
        - Start exactly with: This is classified as {topic} because
        - Use TEXT as the main evidence.
        - Do not continue CONTEXT.
        - Do not mention context.
        - Do not write labels like "Text classified", "Predicted classification", or "Relevant context".
        - Do not include article fragments.
        - Maximum 25 words.
        """).strip()

        reply = self.generate_reply(prompt)

        return self.clean_explanation(reply, topic, extracted_text)

    def clean_explanation(self, reply, topic, extracted_text):
        reply = str(reply).strip()

        banned = [
            "--- END OF ARTICLE ---",
            "-- END OF ARTICLE --",
            "END OF ARTICLE",
            "Relevant context:",
            "Relevant Context:",
            "Text classified:",
            "Text classified",
            "Predicted classification:",
            "Predicted Classification:",
            "User input:",
            "CONTEXT:",
            "TEXT:",
            "PREDICTED_TOPIC:",
            "OUTPUT RULES:",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>"
        ]

        for item in banned:
            reply = reply.replace(item, "").strip()

        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        reply = lines[0] if lines else ""

        first_end = None

        for mark in [".", "!", "?"]:
            idx = reply.find(mark)

            if idx != -1:
                if first_end is None or idx < first_end:
                    first_end = idx

        if first_end is not None:
            reply = reply[:first_end + 1].strip()

        required_start = f"This is classified as {topic} because"

        if not reply.lower().startswith(required_start.lower()):
            return self.fallback_explanation(topic, extracted_text)

        if len(reply.split()) > 30:
            return self.fallback_explanation(topic, extracted_text)

        return reply

    def fallback_explanation(self, topic, extracted_text):
        return (
            f"This is classified as {topic} because the text "
            f"\"{extracted_text}\" contains wording associated with {topic} news."
        )

    def get_input(self):
        return input("You: ")

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
    bot = Chatbot(
        "logistic_regression_topic_model.pkl",
        "news_knowledge"
    )
    bot.run()