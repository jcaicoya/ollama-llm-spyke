import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog, simpledialog
import ollama
import threading
import queue
import datetime
import json
import os 

# --- Constants ---
SYSTEM_PROMPTS_FILE = "system_prompts.json"
PREDEFINED_PROMPTS = {
    "Default": "You are a helpful AI assistant.",
    "Code Explainer": "You are a senior software engineer. Explain code snippets clearly and concisely. Provide examples where appropriate.",
    "Creative Writer": "You are a creative writer. Generate imaginative stories, poems, or scripts based on the user's input.",
    "Fact Checker": "You are a fact-checking AI. Provide accurate information and cite sources if possible.",
    "Summarizer": "You are a text summarizer. Condense provided text into a brief and informative summary.",
    "Spanish Translator": "You are an English to Spanish translator. Translate the given text accurately.",
    "Joke Teller": "You are a comedian. Tell a short, family-friendly joke.",
}

class OllamaChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ollama Local Chat GUI")
        master.geometry("950x850") # Increased size again for new buttons
        master.resizable(True, True)

        self.model_name = tk.StringVar(master)
        self.system_prompt_name = tk.StringVar(master) 
        self.system_prompts = {} 

        self.running_thread = None
        self.response_queue = queue.Queue()
        self.conversation_history = [] # Stores messages for context and saving

        # --- Settings Frame (Model, Temperature, Prompt Controls, Chat Mgmt) ---
        self.settings_frame = ttk.Frame(master, padding="10")
        self.settings_frame.pack(fill=tk.X)

        # Model selection and temperature sub-frame
        self.model_temp_frame = ttk.Frame(self.settings_frame)
        self.model_temp_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.model_temp_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))
        self.model_dropdown = ttk.Combobox(self.model_temp_frame, textvariable=self.model_name, width=25)
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)
        self.refresh_models_button = ttk.Button(self.model_temp_frame, text="Refresh Models", command=self.load_models)
        self.refresh_models_button.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(self.model_temp_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.DoubleVar(value=0.7) 
        self.temperature_scale = ttk.Scale(self.model_temp_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL,
                                            variable=self.temperature_var, command=self.on_temperature_change,
                                            length=120)
        self.temperature_scale.pack(side=tk.LEFT, padx=(5, 5))
        self.temperature_label = ttk.Label(self.model_temp_frame, text="0.70")
        self.temperature_label.pack(side=tk.LEFT)

        # --- System Prompt Management Frame ---
        self.prompt_mgmt_frame = ttk.LabelFrame(master, text="System Prompt Management", padding="10")
        self.prompt_mgmt_frame.pack(padx=10, pady=(0,10), fill=tk.X)

        ttk.Label(self.prompt_mgmt_frame, text="Saved Prompts:").pack(side=tk.LEFT, padx=(0, 5))
        self.prompt_dropdown = ttk.Combobox(self.prompt_mgmt_frame, textvariable=self.system_prompt_name, width=25)
        self.prompt_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.prompt_dropdown.bind("<<ComboboxSelected>>", self.on_prompt_select)

        self.add_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Add New", command=self.add_new_prompt)
        self.add_prompt_button.pack(side=tk.LEFT, padx=(0, 5))
        self.update_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Update", command=self.update_prompt)
        self.update_prompt_button.pack(side=tk.LEFT, padx=(0, 5))
        self.delete_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Delete", command=self.delete_prompt)
        self.delete_prompt_button.pack(side=tk.LEFT, padx=(0, 5))
        self.restore_defaults_button = ttk.Button(self.prompt_mgmt_frame, text="Restore Defaults", command=self.restore_default_prompts)
        self.restore_defaults_button.pack(side=tk.LEFT)

        # --- Current System Prompt Input Area ---
        ttk.Label(master, text="Current System Prompt:").pack(padx=10, anchor=tk.W)
        self.system_prompt_input = scrolledtext.ScrolledText(master, height=4, wrap=tk.WORD, font=("Arial", 10), bg="#E8E8E8")
        self.system_prompt_input.pack(padx=10, pady=(0,10), fill=tk.X)
        
        # --- Chat Management Buttons ---
        self.chat_mgmt_frame = ttk.Frame(master, padding="10")
        self.chat_mgmt_frame.pack(padx=10, pady=(0,10), fill=tk.X)

        self.save_chat_button = ttk.Button(self.chat_mgmt_frame, text="Save Chat", command=self.save_chat)
        self.save_chat_button.pack(side=tk.LEFT, padx=(0, 5))
        self.load_chat_button = ttk.Button(self.chat_mgmt_frame, text="Load Chat", command=self.load_chat)
        self.load_chat_button.pack(side=tk.LEFT, padx=(0, 5))
        self.clear_chat_button = ttk.Button(self.chat_mgmt_frame, text="Clear Chat", command=self.clear_chat_history_display)
        self.clear_chat_button.pack(side=tk.LEFT)


        # --- Conversation History Display ---
        self.chat_history_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, state='disabled', font=("Arial", 10), bg="#F5F5F5")
        self.chat_history_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

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

        # Initialize models and system prompts, then start GUI update loop
        self.load_models()
        self.load_system_prompts()
        self.master.after(100, self.process_queue)
        
        # Select "Default" prompt on startup if available
        if "Default" in self.system_prompts:
            self.system_prompt_name.set("Default")
            self.system_prompt_input.delete("1.0", tk.END)
            self.system_prompt_input.insert(tk.END, self.system_prompts["Default"])


    # --- Model Loading Functions (Unchanged) ---
    def load_models(self):
        self.status_bar.config(text="Fetching models from Ollama...")
        print("Attempting to load models...")
        try:
            models_info = ollama.list()
            print(f"ollama.list() returned: {models_info}")

            model_names = []
            if 'models' in models_info and isinstance(models_info['models'], list):
                for m in models_info['models']:
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
                if 'llama3.1:8b' in model_names:
                    self.model_name.set('llama3.1:8b')
                elif 'llama3:8b' in model_names:
                    self.model_name.set('llama3:8b')
                else:
                    self.model_name.set(model_names[0])
            self.status_bar.config(text=f"Models loaded. Current model: {self.model_name.get()}")
            print("Models loaded successfully.")

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
        self.conversation_history = [] # Clear history when model changes
        self.clear_chat_history_display() # Clear display too

    def on_temperature_change(self, value):
        self.temperature_label.config(text=f"{float(value):.2f}")

    # --- System Prompt Management Functions (Unchanged) ---
    def load_system_prompts(self):
        try:
            if os.path.exists(SYSTEM_PROMPTS_FILE):
                with open(SYSTEM_PROMPTS_FILE, 'r') as f:
                    self.system_prompts = json.load(f)
                print(f"Loaded system prompts from {SYSTEM_PROMPTS_FILE}")
            else:
                self.system_prompts = dict(PREDEFINED_PROMPTS)
                self.save_system_prompts()
                print(f"System prompts file not found. Loaded and saved default prompts to {SYSTEM_PROMPTS_FILE}")
        except json.JSONDecodeError as e:
            messagebox.showerror("File Error", f"Error reading {SYSTEM_PROMPTS_FILE}: {e}. Restoring default prompts.")
            self.system_prompts = dict(PREDEFINED_PROMPTS)
            self.save_system_prompts()
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred loading prompts: {e}. Restoring default prompts.")
            self.system_prompts = dict(PREDEFINED_PROMPTS)
            self.save_system_prompts()
        
        self.update_prompt_dropdown()

    def save_system_prompts(self):
        try:
            with open(SYSTEM_PROMPTS_FILE, 'w') as f:
                json.dump(self.system_prompts, f, indent=4)
            print(f"System prompts saved to {SYSTEM_PROMPTS_FILE}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save system prompts: {e}")

    def update_prompt_dropdown(self):
        self.prompt_dropdown['values'] = sorted(list(self.system_prompts.keys()))
        if self.system_prompt_name.get() not in self.system_prompts and self.system_prompts:
            self.system_prompt_name.set(sorted(list(self.system_prompts.keys()))[0])
            self.on_prompt_select(None)
        elif not self.system_prompts:
            self.system_prompt_name.set("")
            self.system_prompt_input.delete("1.0", tk.END)

    def on_prompt_select(self, event):
        selected_name = self.system_prompt_name.get()
        if selected_name in self.system_prompts:
            self.system_prompt_input.delete("1.0", tk.END)
            self.system_prompt_input.insert(tk.END, self.system_prompts[selected_name])
            self.status_bar.config(text=f"Loaded system prompt: '{selected_name}'")
        else:
            self.system_prompt_input.delete("1.0", tk.END)
            self.status_bar.config(text="No system prompt selected or found.")

    def add_new_prompt(self):
        new_name = simpledialog.askstring("Add New Prompt", "Enter a name for the new system prompt:")
        if new_name and new_name.strip():
            new_name = new_name.strip()
            if new_name in self.system_prompts:
                response = messagebox.askyesno("Prompt Exists", f"Prompt '{new_name}' already exists. Do you want to update it?")
                if not response:
                    return
            
            self.system_prompts[new_name] = self.system_prompt_input.get("1.0", tk.END).strip()
            self.save_system_prompts()
            self.update_prompt_dropdown()
            self.system_prompt_name.set(new_name)
            self.status_bar.config(text=f"Added/Updated prompt: '{new_name}'")
        else:
            messagebox.showwarning("Invalid Name", "Prompt name cannot be empty.")

    def update_prompt(self):
        current_name = self.system_prompt_name.get()
        if not current_name:
            messagebox.showwarning("No Prompt Selected", "Please select a prompt to update.")
            return

        if current_name in self.system_prompts:
            self.system_prompts[current_name] = self.system_prompt_input.get("1.0", tk.END).strip()
            self.save_system_prompts()
            self.status_bar.config(text=f"Updated prompt: '{current_name}'")
        else:
            messagebox.showwarning("Prompt Not Found", f"Prompt '{current_name}' not found. Consider adding it as new.")

    def delete_prompt(self):
        current_name = self.system_prompt_name.get()
        if not current_name:
            messagebox.showwarning("No Prompt Selected", "Please select a prompt to delete.")
            return
        
        if current_name == "Default":
            messagebox.showwarning("Cannot Delete Default", "The 'Default' system prompt cannot be deleted.")
            return

        if current_name in self.system_prompts:
            response = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete prompt '{current_name}'?")
            if response:
                del self.system_prompts[current_name]
                self.save_system_prompts()
                self.update_prompt_dropdown()
                self.system_prompt_name.set("Default")
                self.on_prompt_select(None)
                self.status_bar.config(text=f"Deleted prompt: '{current_name}'")
        else:
            messagebox.showwarning("Prompt Not Found", f"Prompt '{current_name}' not found.")

    def restore_default_prompts(self):
        response = messagebox.askyesno("Restore Defaults", "Are you sure you want to restore all default system prompts? This will overwrite any custom prompts.")
        if response:
            self.system_prompts = dict(PREDEFINED_PROMPTS)
            self.save_system_prompts()
            self.update_prompt_dropdown()
            self.system_prompt_name.set("Default")
            self.on_prompt_select(None)
            self.status_bar.config(text="Restored default system prompts.")

    # --- Chat History Management Functions ---
    def display_full_chat_history(self):
        """Redraws the entire chat display from self.conversation_history."""
        self.chat_history_display.config(state='normal')
        self.chat_history_display.delete("1.0", tk.END) # Clear existing text
        
        # We need to explicitly display the system prompt if it's active
        system_prompt = self.system_prompt_input.get("1.0", tk.END).strip()
        if system_prompt:
            self.chat_history_display.insert(tk.END, f"[Current System Prompt]:\n", ("sender_tag",))
            self.chat_history_display.insert(tk.END, f"{system_prompt}\n\n", ("system_tag",))
            self.chat_history_display.tag_config("system_tag", foreground="purple", font=("Arial", 10, "italic")) # New tag for system prompt display

        for msg in self.conversation_history:
            sender_display = "You" if msg['role'] == 'user' else "Model"
            color = "blue" if msg['role'] == 'user' else "black"
            # Note: We don't have timestamps for loaded history, so we omit them for loaded messages
            self.chat_history_display.insert(tk.END, f"{sender_display}:\n", ("sender_tag",))
            self.chat_history_display.insert(tk.END, f"{msg['content']}\n\n")
            
        self.chat_history_display.config(state='disabled')
        self.chat_history_display.see(tk.END) # Scroll to the end
        self.chat_history_display.tag_config("sender_tag", foreground=color, font=("Arial", 10, "bold")) # Reapply tag config

    def clear_chat_history_display(self):
        """Clears the displayed chat history and internal history."""
        response = messagebox.askyesno("Clear Chat", "Are you sure you want to clear the current chat?")
        if response:
            self.conversation_history = []
            self.chat_history_display.config(state='normal')
            self.chat_history_display.delete("1.0", tk.END)
            self.chat_history_display.config(state='disabled')
            self.status_bar.config(text="Chat history cleared.")

    def save_chat(self):
        """Saves the current conversation history to a JSON file."""
        if not self.conversation_history:
            messagebox.showinfo("No Chat", "There is no conversation to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Chat As",
            initialfile=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if file_path:
            try:
                # Get the currently active system prompt too
                current_system_prompt = self.system_prompt_input.get("1.0", tk.END).strip()
                
                # Store chat history along with the system prompt used for context
                data_to_save = {
                    "system_prompt_used": current_system_prompt,
                    "model_used": self.model_name.get(),
                    "temperature_used": self.temperature_var.get(),
                    "conversation_history": self.conversation_history
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=4, ensure_ascii=False)
                self.status_bar.config(text=f"Chat saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save chat: {e}")

    def load_chat(self):
        """Loads conversation history from a JSON file."""
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Chat"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                # Optional: Ask user if they want to clear current chat
                if self.conversation_history:
                    response = messagebox.askyesno("Load Chat", "Loading a new chat will clear the current conversation. Continue?")
                    if not response:
                        return

                # Load conversation history
                self.conversation_history = loaded_data.get("conversation_history", [])
                
                # Optional: Restore model, temperature, and system prompt from saved data
                loaded_model = loaded_data.get("model_used")
                if loaded_model and loaded_model in self.model_dropdown['values']:
                    self.model_name.set(loaded_model)
                    self.status_bar.config(text=f"Loaded model: {loaded_model}")
                else:
                    self.status_bar.config(text="Warning: Original model not found or selected for loaded chat.")
                    
                loaded_temp = loaded_data.get("temperature_used")
                if loaded_temp is not None:
                    self.temperature_var.set(loaded_temp)
                    self.on_temperature_change(loaded_temp) # Update label
                    self.status_bar.config(text=f"{self.status_bar.cget('text')} | Loaded temp: {loaded_temp:.2f}")

                loaded_system_prompt = loaded_data.get("system_prompt_used", "")
                self.system_prompt_input.delete("1.0", tk.END)
                self.system_prompt_input.insert(tk.END, loaded_system_prompt)
                self.system_prompt_name.set("") # Clear selection as it's custom loaded
                self.status_bar.config(text=f"{self.status_bar.cget('text')} | Loaded system prompt.")

                self.display_full_chat_history() # Redraw the chat display
                self.status_bar.config(text=f"Chat loaded from {os.path.basename(file_path)}")
            except json.JSONDecodeError as e:
                messagebox.showerror("Load Error", f"Failed to read chat file (invalid JSON): {e}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load chat: {e}")

    # --- Message Display and Sending Functions ---
    def display_message(self, sender, message, color="black"):
        self.chat_history_display.config(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_history_display.insert(tk.END, f"[{timestamp}] {sender}:\n", ("sender_tag",))
        self.chat_history_display.insert(tk.END, f"{message}\n\n")
        self.chat_history_display.config(state='disabled')
        self.chat_history_display.see(tk.END)
        self.chat_history_display.tag_config("sender_tag", foreground=color, font=("Arial", 10, "bold"))

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

        system_prompt = self.system_prompt_input.get("1.0", tk.END).strip()
        temperature = self.temperature_var.get()

        self.display_message("You", user_text, color="blue")
        self.user_input.delete("1.0", tk.END)

        messages_to_send = []
        if system_prompt:
            messages_to_send.append({'role': 'system', 'content': system_prompt})
        messages_to_send.extend(self.conversation_history)
        messages_to_send.append({'role': 'user', 'content': user_text})


        self.status_bar.config(text=f"Generating response from {current_model}...")
        self.send_button.config(state=tk.DISABLED)
        self.user_input.config(state=tk.DISABLED)
        self.model_dropdown.config(state=tk.DISABLED)
        self.temperature_scale.config(state=tk.DISABLED)
        self.system_prompt_input.config(state=tk.DISABLED)
        self.add_prompt_button.config(state=tk.DISABLED)
        self.update_prompt_button.config(state=tk.DISABLED)
        self.delete_prompt_button.config(state=tk.DISABLED)
        self.restore_defaults_button.config(state=tk.DISABLED)
        self.prompt_dropdown.config(state=tk.DISABLED)
        self.save_chat_button.config(state=tk.DISABLED) # Disable chat management
        self.load_chat_button.config(state=tk.DISABLED)
        self.clear_chat_button.config(state=tk.DISABLED)


        self.running_thread = threading.Thread(target=self._get_llm_response,
                                               args=(messages_to_send, current_model, temperature, user_text))
        self.running_thread.start()

    def _get_llm_response(self, messages, model, temperature, user_text):
        current_ai_response = ""
        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                options={'temperature': temperature},
                stream=True
            )
            self.response_queue.put(('start_response', "Model"))
            for chunk in stream:
                token = chunk['message']['content']
                if token:
                    self.response_queue.put(('token', token))
                    current_ai_response += token
            
            self.response_queue.put(('add_to_history', {'role': 'user', 'content': user_text}))
            self.response_queue.put(('add_to_history', {'role': 'assistant', 'content': current_ai_response}))

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
                    self.chat_history_display.config(state='normal')
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    self.chat_history_display.insert(tk.END, f"[{timestamp}] {data}:\n", ("sender_tag",))
                    self.chat_history_display.config(state='disabled')
                    self.chat_history_display.see(tk.END)
                elif task_type == 'token':
                    self.chat_history_display.config(state='normal')
                    self.chat_history_display.insert(tk.END, data)
                    self.chat_history_display.config(state='disabled')
                    self.chat_history_display.see(tk.END)
                elif task_type == 'end_response':
                    self.chat_history_display.config(state='normal')
                    self.chat_history_display.insert(tk.END, "\n\n")
                    self.chat_history_display.config(state='disabled')
                    self.chat_history_display.see(tk.END)
                    self.status_bar.config(text=f"Response complete for {self.model_name.get()}")
                    self.send_button.config(state=tk.NORMAL)
                    self.user_input.config(state=tk.NORMAL)
                    self.model_dropdown.config(state=tk.NORMAL)
                    self.temperature_scale.config(state=tk.NORMAL)
                    self.system_prompt_input.config(state=tk.NORMAL)
                    self.add_prompt_button.config(state=tk.NORMAL)
                    self.update_prompt_button.config(state=tk.NORMAL)
                    self.delete_prompt_button.config(state=tk.NORMAL)
                    self.restore_defaults_button.config(state=tk.NORMAL)
                    self.prompt_dropdown.config(state=tk.NORMAL)
                    self.save_chat_button.config(state=tk.NORMAL) # Re-enable chat management
                    self.load_chat_button.config(state=tk.NORMAL)
                    self.clear_chat_button.config(state=tk.NORMAL)

                elif task_type == 'add_to_history':
                    self.conversation_history.append(data)
                elif task_type == 'error':
                    messagebox.showerror("Error During Generation", data)
                    self.status_bar.config(text="Error during generation.")
                    self.send_button.config(state=tk.NORMAL)
                    self.user_input.config(state=tk.NORMAL)
                    self.model_dropdown.config(state=tk.NORMAL)
                    self.temperature_scale.config(state=tk.NORMAL)
                    self.system_prompt_input.config(state=tk.NORMAL)
                    self.add_prompt_button.config(state=tk.NORMAL)
                    self.update_prompt_button.config(state=tk.NORMAL)
                    self.delete_prompt_button.config(state=tk.NORMAL)
                    self.restore_defaults_button.config(state=tk.NORMAL)
                    self.prompt_dropdown.config(state=tk.NORMAL)
                    self.save_chat_button.config(state=tk.NORMAL) # Re-enable chat management
                    self.load_chat_button.config(state=tk.NORMAL)
                    self.clear_chat_button.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        self.master.after(100, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatGUI(root)
    root.mainloop()