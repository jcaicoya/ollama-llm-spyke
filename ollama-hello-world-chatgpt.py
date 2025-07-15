import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import ollama
import threading
import queue
import json
import os

client = ollama.Client()


class OllamaChatApp:
    PROMPTS_FILE = "prompts.json"
    CONFIG_FILE = "config.json"

    DEFAULT_PROMPTS = {
        "Helpful assistant": "You are a helpful assistant.",
        "Sarcastic friend": "You are a sarcastic, witty companion.",
        "Technical expert": "You are a technical expert who explains concisely.",
        "Custom": ""
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Ollama LLM Chat")

        # ==== Top controls frame ====
        controls_frame = tk.Frame(root)
        controls_frame.pack(pady=(10, 0))

        tk.Label(controls_frame, text="Model:").grid(row=0, column=0, padx=5, sticky="w")
        self.model_var = tk.StringVar(value="llama3.1:8b")
        self.models = ["llama3.1:8b", "mistral:7b", "llama2:7b", "llama3:instruct"]
        self.model_menu = ttk.Combobox(
            controls_frame, textvariable=self.model_var, values=self.models, state="readonly", width=20
        )
        self.model_menu.grid(row=0, column=1, padx=5)

        tk.Label(controls_frame, text="Temperature:").grid(row=0, column=2, padx=5, sticky="w")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_slider = tk.Scale(
            controls_frame, variable=self.temperature_var, from_=0.0, to=1.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=150
        )
        self.temperature_slider.grid(row=0, column=3, padx=5)

        tk.Label(controls_frame, text="System prompt:").grid(row=1, column=0, padx=5, sticky="w")
        self.system_prompt_choice_var = tk.StringVar(value="Helpful assistant")
        self.system_prompt_text_var = tk.StringVar()

        self.system_prompt_menu = ttk.Combobox(
            controls_frame,
            textvariable=self.system_prompt_choice_var,
            state="readonly",
            width=20
        )
        self.system_prompt_menu.grid(row=1, column=1, padx=5, sticky="w")

        self.system_prompt_entry = tk.Entry(
            controls_frame,
            textvariable=self.system_prompt_text_var,
            width=70
        )
        self.system_prompt_entry.grid(row=1, column=2, columnspan=2, padx=5, pady=(5, 10), sticky="w")

        # ==== Paned window ====
        self.paned_window = tk.PanedWindow(root, orient=tk.VERTICAL, sashwidth=4, sashpad=2)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.chat_area = scrolledtext.ScrolledText(
            self.paned_window, wrap=tk.WORD, state="disabled"
        )
        self.paned_window.add(self.chat_area, minsize=100)

        self.log_area = scrolledtext.ScrolledText(
            self.paned_window, wrap=tk.WORD, state="disabled", background="#f0f0f0"
        )
        self.paned_window.add(self.log_area, minsize=50)

        self.log_visible = True

        # ==== Entry field ====
        entry_frame = tk.Frame(root)
        entry_frame.pack(pady=(0, 10))

        self.entry = tk.Entry(entry_frame, width=50)
        self.entry.pack(side=tk.LEFT, padx=(10, 0))
        self.entry.bind("<Return>", self.on_send)

        tk.Button(entry_frame, text="Send", command=self.on_send).pack(side=tk.LEFT, padx=5)
        tk.Button(entry_frame, text="Save Chat", command=self.save_chat).pack(side=tk.LEFT, padx=5)
        tk.Button(entry_frame, text="Load Chat", command=self.load_chat).pack(side=tk.LEFT, padx=5)
        tk.Button(entry_frame, text="Load Log", command=self.load_log_file).pack(side=tk.LEFT, padx=5)

        self.toggle_log_button = tk.Button(entry_frame, text="Hide Log", command=self.toggle_log_view)
        self.toggle_log_button.pack(side=tk.LEFT, padx=5)

        # ==== State ====
        self.messages = []
        self.response_queue = queue.Queue()
        self.poll_response_queue()
        self.load_prompts()
        self.config = self.load_config()

        # Restore last log and sash
        last_log = self.config.get("last_log_file_path")
        if last_log and os.path.exists(last_log):
            self.load_log_file(last_log)
        if "sash_position" in self.config:
            self.root.after(100, lambda: self.paned_window.sash_place(0, 0, self.config["sash_position"]))

        # Save sash on resize
        self.root.bind("<Configure>", self.on_resize)

    # ==== Chat logic ====

    def on_send(self, event=None):
        prompt = self.entry.get().strip()
        if not prompt:
            messagebox.showwarning("Empty input", "Please type a question.")
            return
        self.entry.delete(0, tk.END)
        self.append_message(f"🧑‍💻 You: {prompt}\n")
        self.messages.append({"role": "user", "content": prompt})
        threading.Thread(target=self.get_response, daemon=True).start()

    def get_response(self):
        full_response = ""
        model = self.model_var.get()
        temperature = self.temperature_var.get()
        system_prompt = self.system_prompt_text_var.get().strip()

        if not any(m["role"] == "system" for m in self.messages):
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        self.response_queue.put("🤖 Ollama: ")

        try:
            stream = client.chat(
                model=model,
                messages=self.messages,
                options={"temperature": temperature},
                stream=True,
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                self.response_queue.put(token)

        except Exception as e:
            full_response = f"\n⚠️ Error: {e}\n"
            self.response_queue.put(full_response)

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

    # ==== Prompts ====

    def load_prompts(self):
        if os.path.exists(self.PROMPTS_FILE):
            with open(self.PROMPTS_FILE, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        else:
            prompts = self.DEFAULT_PROMPTS.copy()
        self.system_prompts = prompts
        self.update_prompt_menu()

    def update_prompt_menu(self):
        menu_values = list(self.system_prompts.keys())
        if "Custom" not in menu_values:
            menu_values.append("Custom")
        self.system_prompt_menu["values"] = menu_values
        self.system_prompt_choice_var.set("Helpful assistant")
        self.system_prompt_text_var.set(self.system_prompts["Helpful assistant"])
        self.system_prompt_menu.bind("<<ComboboxSelected>>", self.on_system_prompt_selected)

    def on_system_prompt_selected(self, event=None):
        choice = self.system_prompt_choice_var.get()
        self.system_prompt_text_var.set(self.system_prompts.get(choice, ""))

    # ==== Chat & log history ====

    def save_chat(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        with open(path, "w") as f:
            f.write(self.chat_area.get(1.0, tk.END))
        messagebox.showinfo("Saved", f"Chat saved to {path}.")

    def load_chat(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        with open(path, "r") as f:
            content = f.read()
        self.chat_area.config(state="normal")
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.insert(tk.END, content)
        self.chat_area.config(state="disabled")

    def load_log_file(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("Log files", "*.log *.txt")])
            if not path:
                return
        if not os.path.exists(path):
            messagebox.showwarning("Log File", f"Log file '{path}' does not exist.")
            return
        with open(path, "r") as f:
            lines = f.readlines()
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, tk.END)
        self.log_area.insert(tk.END, "\n".join([line.strip() for line in lines]))
        self.log_area.config(state="disabled")
        self.config["last_log_file_path"] = path
        self.save_config()
        if not self.log_visible:
            self.toggle_log_view()

    def toggle_log_view(self):
        if self.log_visible:
            self.paned_window.forget(self.log_area)
            self.toggle_log_button.config(text="Show Log")
        else:
            self.paned_window.add(self.log_area)
            self.toggle_log_button.config(text="Hide Log")
        self.log_visible = not self.log_visible

    def on_resize(self, event=None):
        try:
            pos = self.paned_window.sash_coord(0)[1]
            self.config["sash_position"] = pos
            self.save_config()
        except Exception:
            pass

    # ==== Config ====

    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_config(self):
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatApp(root)
    root.mainloop()
