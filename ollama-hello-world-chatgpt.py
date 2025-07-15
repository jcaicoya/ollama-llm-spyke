import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import ollama
import threading
import queue

client = ollama.Client()


class OllamaChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama LLM Chat")

        # Model selector
        model_frame = tk.Frame(root)
        model_frame.pack(pady=(10, 0))

        tk.Label(model_frame, text="Select model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_var = tk.StringVar(value="llama3.1:8b")

        # You can extend this list as you pull more models
        self.models = ["llama3.1:8b", "mistral:7b", "llama2:7b", "llama3:instruct"]

        self.model_menu = ttk.Combobox(
            model_frame, textvariable=self.model_var, values=self.models, state="readonly"
        )
        self.model_menu.pack(side=tk.LEFT)

        # Conversation text area
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", height=20, width=80
        )
        self.chat_area.pack(padx=10, pady=10)

        # Entry field
        entry_frame = tk.Frame(root)
        entry_frame.pack(pady=(0, 10))

        self.entry = tk.Entry(entry_frame, width=70)
        self.entry.pack(side=tk.LEFT, padx=(10, 0))
        self.entry.bind("<Return>", self.on_send)

        self.send_button = tk.Button(entry_frame, text="Send", command=self.on_send)
        self.send_button.pack(side=tk.LEFT, padx=10)

        self.messages = []

        # Queue for thread-safe UI updates
        self.response_queue = queue.Queue()
        self.poll_response_queue()

    def on_send(self, event=None):
        prompt = self.entry.get().strip()
        if not prompt:
            messagebox.showwarning("Empty input", "Please type a question.")
            return

        self.entry.delete(0, tk.END)

        # Display user prompt
        self.append_message(f"🧑‍💻 You: {prompt}\n")
        self.messages.append({"role": "user", "content": prompt})

        # Run Ollama call in a thread
        threading.Thread(target=self.get_response, daemon=True).start()

    def get_response(self):
        full_response = ""
        model = self.model_var.get()

        self.response_queue.put("🤖 Ollama: ")

        try:
            stream = client.chat(
                model=model,
                messages=self.messages,
                stream=True,
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                self.response_queue.put(token)

        except Exception as e:
            full_response = f"\n⚠️ Error: {e}\n"
            self.response_queue.put(full_response)

        # Append full response to conversation history
        self.messages.append({"role": "assistant", "content": full_response})
        self.response_queue.put("\n")

    def poll_response_queue(self):
        try:
            while True:
                token = self.response_queue.get_nowait()
                self.chat_area.config(state="normal")
                self.chat_area.insert(tk.END, token)
                self.chat_area.see(tk.END)
                self.chat_area.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(50, self.poll_response_queue)

    def append_message(self, message):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, message)
        self.chat_area.see(tk.END)
        self.chat_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatApp(root)
    root.mainloop()
