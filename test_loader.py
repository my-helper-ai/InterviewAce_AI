from utils.loader import load_document

text = load_document("data/Python/python_interview.pdf")

print(text[:1000])