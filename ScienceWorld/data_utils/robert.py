import os

os.environ["http proxy"]="http://127.0.0.1:7890'"
os.environ["https_proxy"]="http://127.0.0.1:7890"



from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
