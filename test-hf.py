from transformers import AutoTokenizer

print("Attempting to download tokenizer...")
# This is a very small download
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
print("Success!")
