
#hugging face transformer

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_auth_token=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", use_auth_token=False)

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,)

story = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("generated text:\n",story)
