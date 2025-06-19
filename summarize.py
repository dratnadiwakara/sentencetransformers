from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

labeler = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

labeler = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

def generate_issue_label(text):
    prompt = f"generate a short label for the issue: {text}"
    try:
        result = labeler(prompt, max_new_tokens=15, do_sample=False)[0]['generated_text']
        return result.strip().capitalize().rstrip('.')
    except Exception as e:
        return f"Error: {e}"




from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
summarizer = pipeline("summarization", model="t5-small")
def generate_issue_label(text):
    prompt = f"generate a short label for the issue: {text}"
    try:
        result = summarizer(prompt, max_length=15, min_length=4, do_sample=False)
        label = result[0]['summary_text'].strip().rstrip('.')
        return label[0].upper() + label[1:]  # Capitalize first letter
    except:
        return ''

def summarize_issue(text):
    prompt = f"summarize: {text}"
    try:
        return summarizer(prompt, max_length=20, min_length=5, do_sample=False)[0]['summary_text']
    except:
        return ''


def summarize_issue2(text):
    prompt = f"generate a short title for the vehicle issue: {text}"
    try:
        return summarizer(prompt, max_length=15, min_length=3, do_sample=False)[0]['summary_text']
    except:
        return ''
