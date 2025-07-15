import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import ollama
import threading
import queue
import datetime

class OllamaChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ollama Local Chat GUI")
        master.geometry("800x650")
        master.resizable(True, True)

        self.model_name = tk.StringVar(master)
        self.running_thread = None
        self.response_queue = queue.Queue()

        # --- Model Selection Frame ---
        self.model_frame = ttk.Frame(master, padding="10")
        self.model_frame.pack(fill=tk.X)

        ttk.Label(self.model_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))

        self.model_dropdown = ttk.Combobox(self.model_frame, textvariable=self.model_name, width=30)
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)

        self.refresh_models_button = ttk.Button(self.model_frame, text="Refresh Models", command=self.load_models)
        self.refresh_models_button.pack(side=tk.LEFT)

        # --- Conversation History Display ---
        self.chat_history = scrolledtext.ScrolledText(master, wrap=tk.WORD, state='disabled', font=("Arial", 10), bg="#F5F5F5")
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        self.input_frame = ttk.Frame(master, padding="10")
        self.input_frame.pack(fill=tk.X)

        self.user_input = scrolledtext.ScrolledText(self.input_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message_on_enter)

        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        # --- Status Bar ---
        self.status_bar = ttk.Label(master, text="Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize models and start GUI update loop
        self.load_models()
        self.master.after(100, self.process_queue)

    def load_models(self):
        """Loads available Ollama models and populates the dropdown."""
        self.status_bar.config(text="Fetching models from Ollama...")
        print("Attempting to load models...") # Debug print
        try:
            models_info = ollama.list()
            print(f"ollama.list() returned: {models_info}") # Debug print: See the raw output

            model_names = []
            if 'models' in models_info and isinstance(models_info['models'], list):
                for m in models_info['models']:
                    # --- CRUCIAL CHANGE HERE: m.model instead of m.name ---
                    # Check if 'm' has a 'model' attribute and it's a string (the model tag)
                    if hasattr(m, 'model') and isinstance(m.model, str):
                        model_names.append(m.model)
                    else:
                        print(f"Warning: Model item '{m}' does not have a valid 'model' attribute.")
            else:
                print(f"Warning: 'models' key not found or not a list in {models_info}")

            if not model_names:
                messagebox.showwarning("No Models Found",
                                       "No Ollama models found. Please pull a model (e.g., 'ollama pull llama3.1:8b') and ensure Ollama is running.")
                self.model_dropdown['values'] = []
                self.status_bar.config(text="No models loaded.")
                return

            self.model_dropdown['values'] = model_names
            if model_names:
                # Try to pre-select llama3.1:8b if available, otherwise select the first one
                if 'llama3.1:8b' in model_names:
                    self.model_name.set('llama3.1:8b')
                elif 'llama3:8b' in model_names: # Fallback for older Llama 3 models if 3.1 isn't present
                    self.model_name.set('llama3:8b')
                else:
                    self.model_name.set(model_names[0]) # Select the first available model
            self.status_bar.config(text=f"Models loaded. Current model: {self.model_name.get()}")
            print("Models loaded successfully.") # Debug print

        except ollama.ResponseError as e:
            print(f"Ollama Response Error: {e}")
            messagebox.showerror("Ollama Error", f"Failed to connect to Ollama: {e}\nPlease ensure Ollama service is running (`sudo systemctl start ollama`).")
            self.status_bar.config(text="Error connecting to Ollama.")
            self.model_dropdown['values'] = []
        except Exception as e:
            print(f"An unexpected error occurred while loading models: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred while loading models: {e}")
            self.status_bar.config(text="Error loading models.")
            self.model_dropdown['values'] = []


    def on_model_select(self, event):
        selected_model = self.model_name.get()
        self.status_bar.config(text=f"Selected model: {selected_model}")

    def display_message(self, sender, message, color="black"):
        self.chat_history.config(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}:\n", ("sender_tag",))
        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)
        self.chat_history.tag_config("sender_tag", foreground=color, font=("Arial", 10, "bold"))


    def send_message_on_enter(self, event):
        if event.keysym == "Return" and not (event.state & 0x1):
            self.send_message()
            return "break"
        return None


    def send_message(self):
        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text:
            return

        current_model = self.model_name.get()
        if not current_model:
            messagebox.showwarning("No Model Selected", "Please select an Ollama model before sending a message.")
            return

        self.display_message("You", user_text, color="blue")
        self.user_input.delete("1.0", tk.END)

        self.status_bar.config(text=f"Generating response from {current_model}...")
        self.send_button.config(state=tk.DISABLED)
        self.user_input.config(state=tk.DISABLED)
        self.model_dropdown.config(state=tk.DISABLED)

        self.running_thread = threading.Thread(target=self._get_llm_response, args=(user_text, current_model))
        self.running_thread.start()

    def _get_llm_response(self, prompt, model):
        full_response = ""
        try:
            stream = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            self.response_queue.put(('start_response', "Model"))
            for chunk in stream:
                token = chunk['message']['content']
                if token:
                    self.response_queue.put(('token', token))
                    full_response += token
            self.response_queue.put(('end_response', None))
        except ollama.ResponseError as e:
            self.response_queue.put(('error', f"Ollama Error: {e}\nPlease check if model '{model}' is available and Ollama is running."))
        except Exception as e:
            self.response_queue.put(('error', f"An unexpected error occurred: {e}"))


    def process_queue(self):
        try:
            while True:
                task_type, data = self.response_queue.get_nowait()

                if task_type == 'start_response':
                    self.chat_history.config(state='normal')
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    self.chat_history.insert(tk.END, f"[{timestamp}] {data}:\n", ("sender_tag",))
                    self.chat_history.config(state='disabled')
                    self.chat_history.see(tk.END)
                elif task_type == 'token':
                    self.chat_history.config(state='normal')
                    self.chat_history.insert(tk.END, data)
                    self.chat_history.config(state='disabled')
                    self.chat_history.see(tk.END)
                elif task_type == 'end_response':
                    self.chat_history.config(state='normal')
                    self.chat_history.insert(tk.END, "\n\n")
                    self.chat_history.config(state='disabled')
                    self.chat_history.see(tk.END)
                    self.status_bar.config(text=f"Response complete for {self.model_name.get()}")
                    self.send_button.config(state=tk.NORMAL)
                    self.user_input.config(state=tk.NORMAL)
                    self.model_dropdown.config(state=tk.NORMAL)
                elif task_type == 'error':
                    messagebox.showerror("Error During Generation", data)
                    self.status_bar.config(text="Error during generation.")
                    self.send_button.config(state=tk.NORMAL)
                    self.user_input.config(state=tk.NORMAL)
                    self.model_dropdown.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        self.master.after(100, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatGUI(root)
    root.mainloop()