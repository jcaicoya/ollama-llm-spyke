import ollama

def chat_with_llama3_1_streaming(prompt, model_name='llama3.1:8b'):
    """
    Chats with the Llama 3.1 model and streams the response tokens.

    Args:
        prompt (str): The user's input prompt.
        model_name (str): The name of the Ollama model to use (e.g., 'llama3.1:8b', 'llama3.1:8b-q4_K_S').
    """
    print(f"You: {prompt}")
    print(f"{model_name}: ", end="", flush=True) # Prepare for streaming output

    try:
        # The key change is stream=True
        stream = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            stream=True # Enable streaming
        )

        # Iterate over the chunks yielded by the stream
        for chunk in stream:
            # Each chunk contains a 'message' dictionary, and its 'content' is the token
            print(chunk['message']['content'], end='', flush=True)
        print() # Add a newline after the full response
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # --- IMPORTANT: Ensure your Ollama service is running before executing this script ---
    # You can start it with: sudo systemctl start ollama

    # Example usage with streaming
    chat_with_llama3_1_streaming("What are the benefits of using large language models locally?")

    print("\n" + "="*50 + "\n") # Separator for clarity

    chat_with_llama3_1_streaming("Can you explain the concept of quantum entanglement in simple terms?")

    print("\n" + "="*50 + "\n")

    # If you changed your model to a smaller quantization (e.g., 'llama3.1:8b-q4_K_S')
    # you can pass that as the model_name argument:
    # chat_with_llama3_1_streaming("Tell me a short story about a talking cat.", model_name='llama3.1:8b-q4_K_S')
