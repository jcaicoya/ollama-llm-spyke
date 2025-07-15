import ollama

client = ollama.Client()


def chat_stream(model: str, question: str) -> str:
    """
    Stream chat response from Ollama, displaying tokens live and returning full response.
    """
    messages = [{'role': 'user', 'content': question}]
    full_response = ""

    print(f"\n🧑‍💻 Question: {question}\n")
    print("🤖 Answer:\n")

    try:
        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            token = chunk['message']['content']
            full_response += token
            print(token, end='', flush=True)

        print()  # newline

    except Exception as e:
        print(f"\n⚠️ Error while generating response: {e}")

    return full_response


if __name__ == "__main__":
    question = "Tell me a fun fact about space!"
    response = chat_stream(
        model='llama3.1:8b',
        question=question
    )

    # ✅ response is stored, but not printed again

