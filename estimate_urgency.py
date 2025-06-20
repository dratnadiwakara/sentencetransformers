from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Prompt-based scoring function
def estimate_urgency(text):
    prompt = f"Rate the urgency of the following issue on a scale from 0 (not urgent) to 5 (extremely urgent): {text}"
    try:
        response = pipeline(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text']
        score = int(next(word for word in response.split() if word.isdigit()))
        return min(max(score, 0), 5)  # Clamp to [0,5]
    except:
        return None


def classify_tone(text):
    prompt = (
        f"Classify the tone of this statement as one of the following: "
        f"friendly, neutral, confrontational, formal, or demanding. "
        f"Respond with one word.\n\n{text}"
    )
    try:
        response = pipeline(prompt, max_new_tokens=5, do_sample=False)[0]['generated_text']
        return response.strip().lower()
    except:
        return None


import spacy
import re

nlp = spacy.load("en_core_web_sm")

# Detect deadline-like temporal expressions
def has_generic_deadline(text):
    doc = nlp(text)

    # Check for DATE or TIME entities
    for ent in doc.ents:
        if ent.label_ in {'DATE', 'TIME'}:
            # Check if the context is directive or constrained
            for token in ent.root.head.subtree:
                if token.lemma_ in {'complete', 'submit', 'respond', 'implement', 'fix', 'due', 'must', 'should'}:
                    return True

    # Backup: match generic deadline phrases
    patterns = [
        r'\bwithin\s+\d+\s+(days?|weeks?|months?)\b',
        r'\bby\s+\w+\s+\d{1,2}(st|nd|rd|th)?\b',
        r'\bno later than\b',
        r'\bdeadline\b',
        r'\bdue\b'
    ]
    for pattern in patterns:
        if re.search(pattern, text.lower()):
            return True

    return False


import textstat

def readability_score(text):
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return None


import re

def clean_whitespace(text):
    if not isinstance(text, str):
        return text
    # Replace \n, \r, \t with space and remove redundant spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
