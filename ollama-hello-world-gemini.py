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
APP_CONFIG_FILE = "config.json"
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
        # We will set geometry dynamically later, but can leave a default here.
        master.geometry("1100x900")
        master.resizable(True, True)

        # --- Application State Variables ---
        self.model_name = tk.StringVar(master)
        self.system_prompt_name = tk.StringVar(master)
        self.system_prompts = {}

        self.running_thread = None
        self.response_queue = queue.Queue()
        self.conversation_history = []

        self.last_loaded_log_path = tk.StringVar(master)
        self.log_content = ""
        self.log_view_visible = tk.BooleanVar(master, value=True)

        # --- GUI Layout ---
        self.top_frame = ttk.Frame(master, padding="10")
        self.top_frame.pack(fill=tk.X)

        # Model & Settings
        self.model_temp_frame = ttk.LabelFrame(self.top_frame, text="Model & Settings", padding="5")
        self.model_temp_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(self.model_temp_frame, text="Select Model:").pack(side=tk.TOP, anchor=tk.W, pady=(0,5))
        self.model_dropdown = ttk.Combobox(self.model_temp_frame, textvariable=self.model_name, width=25, state='readonly')
        self.model_dropdown.pack(side=tk.TOP, anchor=tk.W, padx=(0, 10), pady=(0, 5))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)
        self.refresh_models_button = ttk.Button(self.model_temp_frame, text="Refresh Models", command=self.load_models)
        self.refresh_models_button.pack(side=tk.TOP, anchor=tk.W, pady=(0, 10))
        ttk.Label(self.model_temp_frame, text="Temperature:").pack(side=tk.TOP, anchor=tk.W)
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_scale = ttk.Scale(self.model_temp_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL, variable=self.temperature_var, command=self.on_temperature_change, length=150)
        self.temperature_scale.pack(side=tk.TOP, anchor=tk.W, padx=(5, 5))
        self.temperature_label = ttk.Label(self.model_temp_frame, text="0.70")
        self.temperature_label.pack(side=tk.TOP, anchor=tk.W)

        # System Prompt Management
        self.prompt_mgmt_frame = ttk.LabelFrame(self.top_frame, text="System Prompt Management", padding="5")
        self.prompt_mgmt_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(self.prompt_mgmt_frame, text="Saved Prompts:").pack(side=tk.TOP, anchor=tk.W, pady=(0,5))
        self.prompt_dropdown = ttk.Combobox(self.prompt_mgmt_frame, textvariable=self.system_prompt_name, width=25, state='readonly')
        self.prompt_dropdown.pack(side=tk.TOP, anchor=tk.W, padx=(0, 10), pady=(0, 5))
        self.prompt_dropdown.bind("<<ComboboxSelected>>", self.on_prompt_select)
        self.add_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Add New", command=self.add_new_prompt)
        self.add_prompt_button.pack(side=tk.TOP, anchor=tk.W, pady=(2, 2))
        self.update_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Update", command=self.update_prompt)
        self.update_prompt_button.pack(side=tk.TOP, anchor=tk.W, pady=(2, 2))
        self.delete_prompt_button = ttk.Button(self.prompt_mgmt_frame, text="Delete", command=self.delete_prompt)
        self.delete_prompt_button.pack(side=tk.TOP, anchor=tk.W, pady=(2, 2))
        self.restore_defaults_button = ttk.Button(self.prompt_mgmt_frame, text="Restore Defaults", command=self.restore_default_prompts)
        self.restore_defaults_button.pack(side=tk.TOP, anchor=tk.W, pady=(2, 2))

        # Chat & Log Management
        self.chat_log_mgmt_frame = ttk.LabelFrame(self.top_frame, text="Chat & Log Management", padding="5")
        self.chat_log_mgmt_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.save_chat_button = ttk.Button(self.chat_log_mgmt_frame, text="Save Chat", command=self.save_chat)
        self.save_chat_button.pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))
        self.load_chat_button = ttk.Button(self.chat_log_mgmt_frame, text="Load Chat", command=self.load_chat)
        self.load_chat_button.pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))
        self.clear_chat_button = ttk.Button(self.chat_log_mgmt_frame, text="Clear Chat", command=self.clear_chat_session)
        self.clear_chat_button.pack(side=tk.TOP, anchor=tk.W, pady=(0, 10))
        ttk.Label(self.chat_log_mgmt_frame, text="Log File:").pack(side=tk.TOP, anchor=tk.W)
        self.last_log_path_label = ttk.Label(self.chat_log_mgmt_frame, textvariable=self.last_loaded_log_path, wraplength=180, font=("Arial", 8))
        self.last_log_path_label.pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))
        self.load_log_button = ttk.Button(self.chat_log_mgmt_frame, text="Load Log File", command=self.load_log_file)
        self.load_log_button.pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))
        self.toggle_log_button = ttk.Checkbutton(self.chat_log_mgmt_frame, text="Show Log View", variable=self.log_view_visible, command=self.toggle_log_view, onvalue=True, offvalue=False)
        self.toggle_log_button.pack(side=tk.TOP, anchor=tk.W)

        # --- Main Content Area: PanedWindow ---
        self.main_content_frame = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Pane
        self.left_column_frame = ttk.Frame(self.main_content_frame)
        ttk.Label(self.left_column_frame, text="Current System Prompt:").pack(padx=0, anchor=tk.W)
        self.system_prompt_input = scrolledtext.ScrolledText(self.left_column_frame, height=4, wrap=tk.WORD, font=("Arial", 10))
        self.system_prompt_input.pack(padx=0, pady=(0,10), fill=tk.X)
        ttk.Label(self.left_column_frame, text="Loaded Log File Content:").pack(padx=0, anchor=tk.W)
        self.log_display = scrolledtext.ScrolledText(self.left_column_frame, wrap=tk.WORD, state='disabled', font=("Courier New", 9), bg="#F8F8F8")
        self.log_display.pack(padx=0, pady=(0,10), fill=tk.BOTH, expand=True)

        # Right Pane
        self.right_column_frame = ttk.Frame(self.main_content_frame)
        self.chat_history_display = scrolledtext.ScrolledText(self.right_column_frame, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.chat_history_display.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)
        self.input_frame = ttk.Frame(self.right_column_frame, padding="10")
        self.input_frame.pack(fill=tk.X)
        self.user_input = scrolledtext.ScrolledText(self.input_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message_on_enter)
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        # Status Bar
        self.status_bar = ttk.Label(master, text="Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Tag Configurations ---
        self.chat_history_display.tag_config("user_tag", foreground="navy", font=("Arial", 10, "bold"))
        self.chat_history_display.tag_config("model_tag", foreground="#006400", font=("Arial", 10, "bold")) # Dark Green
        self.chat_history_display.tag_config("error_tag", foreground="red", font=("Arial", 10, "bold"))

        # --- Initialization ---
        self.load_app_config()
        self.load_models()
        self.load_system_prompts()
        self.main_content_frame.add(self.left_column_frame, weight=1)
        self.main_content_frame.add(self.right_column_frame, weight=2)
        self.toggle_log_view()
        if self.last_loaded_log_path.get() and self.log_view_visible.get():
            self._load_and_display_log_file(self.last_loaded_log_path.get())
        elif self.last_loaded_log_path.get():
             self.last_log_path_label.config(text=os.path.basename(self.last_loaded_log_path.get()))
        self.master.after(100, self.process_queue)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        # ADDED: Set the initial size dynamically to ensure all widgets are visible.
        self._set_initial_size()

    def on_closing(self):
        """Handle window closing event."""
        self.save_app_config()
        self.master.destroy()

    # ADDED: New method to calculate and set the window's optimal initial size.
    def _set_initial_size(self):
        """Calculates and sets the minimum and initial size of the window."""
        # Force tkinter to render the widgets and calculate their required size
        self.master.update_idletasks()

        # Get the minimum required width and height
        min_width = self.master.winfo_reqwidth()
        min_height = self.master.winfo_reqheight()

        # Enforce this as the minimum size of the window
        self.master.minsize(min_width, min_height)

        # Optionally, set the initial size to be comfortable but not smaller than required.
        # The original 1100x900 is a good starting point.
        self.master.geometry(f"{max(1100, min_width)}x{max(900, min_height)}")


    def load_app_config(self):
        try:
            if os.path.exists(APP_CONFIG_FILE):
                with open(APP_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.last_loaded_log_path.set(config.get("last_log_file_path", ""))
                    self.log_view_visible.set(config.get("log_view_visible", True))
            else:
                self.log_view_visible.set(True)
        except (json.JSONDecodeError, IOError) as e:
            messagebox.showerror("Config Error", f"Error reading {APP_CONFIG_FILE}: {e}. Starting with default settings.", parent=self.master)

    def save_app_config(self):
        config = {
            "last_log_file_path": self.last_loaded_log_path.get(),
            "log_view_visible": self.log_view_visible.get()
        }
        try:
            with open(APP_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Warning: Could not save config file: {e}")

    def load_models(self):
        self.status_bar.config(text="Fetching models from Ollama...")
        try:
            models_info = ollama.list()
            model_names = []
            if 'models' in models_info and isinstance(models_info['models'], list):
                for m in models_info['models']:
                    if 'model' in m and isinstance(m['model'], str):
                        model_names.append(m['model'])

            if not model_names:
                messagebox.showwarning("No Models Found", "No Ollama models found. Please pull a model (e.g., 'ollama pull llama3.1:8b') and ensure Ollama is running.", parent=self.master)
                self.model_dropdown['values'] = []
                self.model_name.set("")
                self.status_bar.config(text="No models loaded.")
                return

            self.model_dropdown['values'] = model_names
            preferred_models = ['llama3.1:8b', 'llama3:8b']
            current_selection = self.model_name.get()
            
            if current_selection in model_names:
                return

            for model in preferred_models:
                if model in model_names:
                    self.model_name.set(model)
                    break
            else:
                self.model_name.set(model_names[0])
            self.status_bar.config(text=f"Models loaded. Current model: {self.model_name.get()}")
        except ollama.ResponseError as e:
            messagebox.showerror("Ollama Error", f"Failed to connect to Ollama: {e}\nPlease ensure Ollama service is running.", parent=self.master)
            self.status_bar.config(text="Error connecting to Ollama.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred while loading models: {e}", parent=self.master)
            self.status_bar.config(text="Error loading models.")

    def on_model_select(self, event):
        self.status_bar.config(text=f"Selected model: {self.model_name.get()}. Chat history will be cleared on next message.")
        self.conversation_history = []
        self.clear_chat_display()

    def on_temperature_change(self, value):
        self.temperature_label.config(text=f"{float(value):.2f}")

    def load_system_prompts(self):
        try:
            if os.path.exists(SYSTEM_PROMPTS_FILE):
                with open(SYSTEM_PROMPTS_FILE, 'r') as f:
                    self.system_prompts = json.load(f)
            else:
                self.system_prompts = dict(PREDEFINED_PROMPTS)
                self.save_system_prompts()
        except (json.JSONDecodeError, IOError) as e:
            messagebox.showerror("File Error", f"Error reading {SYSTEM_PROMPTS_FILE}: {e}. Restoring defaults.", parent=self.master)
            self.system_prompts = dict(PREDEFINED_PROMPTS)
        self.update_prompt_dropdown()
        if "Default" in self.system_prompts:
            self.system_prompt_name.set("Default")
            self.on_prompt_select(None)

    def save_system_prompts(self):
        try:
            with open(SYSTEM_PROMPTS_FILE, 'w') as f:
                json.dump(self.system_prompts, f, indent=4)
        except IOError as e:
            messagebox.showerror("Save Error", f"Failed to save system prompts: {e}", parent=self.master)

    def update_prompt_dropdown(self):
        self.prompt_dropdown['values'] = sorted(list(self.system_prompts.keys()))
        current_selection = self.system_prompt_name.get()
        if not current_selection or current_selection not in self.system_prompts:
            if self.system_prompts:
                self.system_prompt_name.set(sorted(list(self.system_prompts.keys()))[0])
                self.on_prompt_select(None)
            else:
                self.system_prompt_name.set("")
                self.system_prompt_input.delete("1.0", tk.END)

    def on_prompt_select(self, event):
        selected_name = self.system_prompt_name.get()
        if selected_name in self.system_prompts:
            self.system_prompt_input.delete("1.0", tk.END)
            self.system_prompt_input.insert(tk.END, self.system_prompts[selected_name])
            self.status_bar.config(text=f"Loaded system prompt: '{selected_name}'")

    def add_new_prompt(self):
        new_name = simpledialog.askstring("Add New Prompt", "Enter a name for the new system prompt:", parent=self.master)
        if new_name and new_name.strip():
            new_name = new_name.strip()
            if new_name in self.system_prompts and not messagebox.askyesno("Prompt Exists", f"Prompt '{new_name}' already exists. Overwrite it?", parent=self.master):
                return
            self.system_prompts[new_name] = self.system_prompt_input.get("1.0", tk.END).strip()
            self.save_system_prompts()
            self.update_prompt_dropdown()
            self.system_prompt_name.set(new_name)
            self.status_bar.config(text=f"Added/Updated prompt: '{new_name}'")
        elif new_name is not None:
            messagebox.showwarning("Invalid Name", "Prompt name cannot be empty.", parent=self.master)

    def update_prompt(self):
        current_name = self.system_prompt_name.get()
        if not current_name:
            messagebox.showwarning("No Prompt Selected", "Please select a prompt to update.", parent=self.master)
            return
        self.system_prompts[current_name] = self.system_prompt_input.get("1.0", tk.END).strip()
        self.save_system_prompts()
        self.status_bar.config(text=f"Updated prompt: '{current_name}'")

    def delete_prompt(self):
        current_name = self.system_prompt_name.get()
        if not current_name:
            messagebox.showwarning("No Prompt Selected", "Please select a prompt to delete.", parent=self.master)
            return
        if current_name == "Default":
            messagebox.showwarning("Cannot Delete Default", "The 'Default' system prompt cannot be deleted.", parent=self.master)
            return
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete prompt '{current_name}'?", parent=self.master):
            del self.system_prompts[current_name]
            self.save_system_prompts()
            self.update_prompt_dropdown()
            self.status_bar.config(text=f"Deleted prompt: '{current_name}'")

    def restore_default_prompts(self):
        if messagebox.askyesno("Restore Defaults", "This will replace all current prompts with the application defaults. Are you sure?", parent=self.master):
            self.system_prompts = dict(PREDEFINED_PROMPTS)
            self.save_system_prompts()
            self.update_prompt_dropdown()
            self.system_prompt_name.set("Default")
            self.on_prompt_select(None)
            self.status_bar.config(text="Restored default system prompts.")

    def clear_chat_display(self):
        """Clears only the visual chat display."""
        self.chat_history_display.config(state='normal')
        self.chat_history_display.delete("1.0", tk.END)
        self.chat_history_display.config(state='disabled')

    def clear_chat_session(self):
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the current chat history?", parent=self.master):
            self.conversation_history = []
            self.clear_chat_display()
            self.status_bar.config(text="Chat history cleared.")

    def save_chat(self):
        if not self.conversation_history:
            messagebox.showinfo("No Chat", "There is no conversation to save.", parent=self.master)
            return
        file_path = filedialog.asksaveasfilename(parent=self.master, defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Chat As", initialfile=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if file_path:
            try:
                data_to_save = {
                    "system_prompt_used": self.system_prompt_input.get("1.0", tk.END).strip(),
                    "model_used": self.model_name.get(),
                    "temperature_used": self.temperature_var.get(),
                    "conversation_history": self.conversation_history
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=4, ensure_ascii=False)
                self.status_bar.config(text=f"Chat saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save chat: {e}", parent=self.master)

    def load_chat(self):
        file_path = filedialog.askopenfilename(parent=self.master, defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Load Chat")
        if file_path:
            try:
                if self.conversation_history and not messagebox.askyesno("Load Chat", "Loading a new chat will clear the current conversation. Continue?", parent=self.master):
                    return
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                self.conversation_history = loaded_data.get("conversation_history", [])
                
                status_parts = []
                loaded_model = loaded_data.get("model_used")
                if loaded_model and loaded_model in self.model_dropdown['values']:
                    self.model_name.set(loaded_model)
                    status_parts.append(f"Model: {loaded_model}")
                else:
                    status_parts.append("Model not found")

                loaded_temp = loaded_data.get("temperature_used")
                if loaded_temp is not None:
                    self.temperature_var.set(loaded_temp)
                    self.on_temperature_change(loaded_temp)
                    status_parts.append(f"Temp: {loaded_temp:.2f}")

                self.system_prompt_input.delete("1.0", tk.END)
                self.system_prompt_input.insert(tk.END, loaded_data.get("system_prompt_used", ""))
                self.system_prompt_name.set("")

                self.clear_chat_display()
                for msg in self.conversation_history:
                    self.response_queue.put(('history_message', msg))
                
                self.status_bar.config(text=f"Chat loaded from {os.path.basename(file_path)} | " + " | ".join(status_parts))
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load chat: {e}", parent=self.master)

    def load_log_file(self):
        file_path = filedialog.askopenfilename(parent=self.master, title="Select Log File", filetypes=[("Log files", "*.log *.txt"), ("All files", "*.*")])
        if file_path:
            self._load_and_display_log_file(file_path)

    def _load_and_display_log_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                messagebox.showwarning("File Not Found", f"The log file '{os.path.basename(file_path)}' was not found.", parent=self.master)
                self.log_content = ""
                self.last_loaded_log_path.set("")
                self._update_log_display()
                return
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.log_content = f.read()
            self.last_loaded_log_path.set(file_path)
            self._update_log_display()
            self.status_bar.config(text=f"Log file loaded: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Log Load Error", f"Failed to load log file: {e}", parent=self.master)
            self.log_content = ""
            self.last_loaded_log_path.set("")
            self._update_log_display()

    def _update_log_display(self):
        self.log_display.config(state='normal')
        self.log_display.delete("1.0", tk.END)
        self.log_display.insert(tk.END, self.log_content)
        self.log_display.config(state='disabled')
        self.log_display.see(tk.END)
        base_name = os.path.basename(self.last_loaded_log_path.get()) if self.last_loaded_log_path.get() else "None"
        self.last_log_path_label.config(text=base_name)

    def toggle_log_view(self):
        is_visible = self.log_view_visible.get()
        log_pane_target_width = 350
        if is_visible:
            self.main_content_frame.sashpos(0, log_pane_target_width)
            self.master.geometry("1100x900")
            self.toggle_log_button.config(text="Hide Log View")
            if self.last_loaded_log_path.get() and not self.log_content:
                self._load_and_display_log_file(self.last_loaded_log_path.get())
        else:
            self.main_content_frame.sashpos(0, 0)
            self.master.geometry("700x900")
            self.toggle_log_button.config(text="Show Log View")

    def send_message_on_enter(self, event):
        if event.keysym == "Return" and not (event.state & 0x1):
            self.send_message()
            return "break"
        return None

    def _set_ui_state(self, state):
        """Helper function to enable/disable UI controls."""
        widgets = [
            self.send_button, self.user_input, self.model_dropdown, self.temperature_scale,
            self.system_prompt_input, self.add_prompt_button, self.update_prompt_button,
            self.delete_prompt_button, self.restore_defaults_button, self.prompt_dropdown,
            self.save_chat_button, self.load_chat_button, self.clear_chat_button,
            self.load_log_button, self.toggle_log_button, self.refresh_models_button
        ]
        for widget in widgets:
            if isinstance(widget, ttk.Combobox):
                widget.config(state='readonly' if state == tk.NORMAL else tk.DISABLED)
            else:
                widget.config(state=state)

    def send_message(self):
        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text: return
        current_model = self.model_name.get()
        if not current_model:
            messagebox.showwarning("No Model Selected", "Please select an Ollama model.", parent=self.master)
            return

        self.chat_history_display.config(state='normal')
        self.chat_history_display.insert(tk.END, f"You:\n", ("user_tag",))
        self.chat_history_display.insert(tk.END, f"{user_text}\n\n")
        self.chat_history_display.config(state='disabled')
        self.chat_history_display.see(tk.END)
        self.user_input.delete("1.0", tk.END)

        system_prompt = self.system_prompt_input.get("1.0", tk.END).strip()
        messages_to_send = []
        if system_prompt:
            messages_to_send.append({'role': 'system', 'content': system_prompt})
        messages_to_send.extend(self.conversation_history)
        messages_to_send.append({'role': 'user', 'content': user_text})

        self.status_bar.config(text=f"Generating response from {current_model}...")
        self._set_ui_state(tk.DISABLED)

        self.running_thread = threading.Thread(target=self._get_llm_response,
            args=(messages_to_send, current_model, self.temperature_var.get(), user_text))
        self.running_thread.start()

    def _get_llm_response(self, messages, model, temperature, user_text):
        current_ai_response = ""
        try:
            stream = ollama.chat(model=model, messages=messages, options={'temperature': temperature}, stream=True)
            self.response_queue.put(('start_response', "Model"))
            for chunk in stream:
                token = chunk['message']['content']
                if token:
                    self.response_queue.put(('token', token))
                    current_ai_response += token
            self.response_queue.put(('add_to_history', {'role': 'user', 'content': user_text}))
            self.response_queue.put(('add_to_history', {'role': 'assistant', 'content': current_ai_response}))
        except ollama.ResponseError as e:
            self.response_queue.put(('error', f"Ollama Error: {e}\nCheck if model '{model}' is available and Ollama is running."))
        except Exception as e:
            self.response_queue.put(('error', f"An unexpected error occurred: {e}"))
        finally:
            self.response_queue.put(('end_response', None))

    def process_queue(self):
        try:
            while True:
                task_type, data = self.response_queue.get_nowait()
                self.chat_history_display.config(state='normal')

                if task_type == 'start_response':
                    self.chat_history_display.insert(tk.END, f"{data}:\n", ("model_tag",))
                elif task_type == 'token':
                    self.chat_history_display.insert(tk.END, data)
                elif task_type == 'end_response':
                    self.chat_history_display.insert(tk.END, "\n\n")
                    self.status_bar.config(text=f"Response complete.")
                    self._set_ui_state(tk.NORMAL)
                elif task_type == 'add_to_history':
                    self.conversation_history.append(data)
                elif task_type == 'history_message':
                    tag = "user_tag" if data['role'] == 'user' else "model_tag"
                    sender = "You" if data['role'] == 'user' else "Model"
                    self.chat_history_display.insert(tk.END, f"{sender}:\n", (tag,))
                    self.chat_history_display.insert(tk.END, f"{data['content']}\n\n")
                elif task_type == 'error':
                    self.chat_history_display.insert(tk.END, f"\n\nERROR:\n{data}\n\n", ("error_tag",))
                    self.status_bar.config(text="Error during generation.")
                    self._set_ui_state(tk.NORMAL)
                
                self.chat_history_display.config(state='disabled')
                self.chat_history_display.see(tk.END)
        except queue.Empty:
            pass
        self.master.after(100, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatGUI(root)
    root.mainloop()