# prompt: text classification using llm


from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_text(text):
  """Classifies the input text using a pre-trained text classification model.

  Args:
    text: The text to classify.

  Returns:
    A dictionary containing the label and score of the classification.
  """
  result = classifier(text)[0]
  return result


# Example usage
text = "This is beautiful project"
classification_result = classify_text(text)
print(f"Text: {text}")
print(f"Label: {classification_result['label']}")
print(f"Score: {classification_result['score']}")


text = "This is not a good project"
classification_result = classify_text(text)
print(f"Text: {text}")
print(f"Label: {classification_result['label']}")
print(f"Score: {classification_result['score']}")
