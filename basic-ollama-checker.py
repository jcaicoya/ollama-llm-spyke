import ollama

client = ollama.Client()

stream = client.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': 'how are you doing?'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

print()
