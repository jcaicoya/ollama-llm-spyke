import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, simpledialog, filedialog
import ollama
import threading
import queue
import json
import os

client = ollama.Client()


class OllamaChatApp:
    PROMPTS_FILE = "prompts.json"

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

        # Model selector
        tk.Label(controls_frame, text="Model:").grid(row=0, column=0, padx=5, sticky="w")
        self.model_var = tk.StringVar(value="llama3.1:8b")
        self.models = ["llama3.1:8b", "mistral:7b", "llama2:7b", "llama3:instruct"]
        self.model_menu = ttk.Combobox(
            controls_frame, textvariable=self.model_var, values=self.models, state="readonly", width=20
        )
        self.model_menu.grid(row=0, column=1, padx=5)

        # Temperature slider
        tk.Label(controls_frame, text="Temperature:").grid(row=0, column=2, padx=5, sticky="w")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_slider = tk.Scale(
            controls_frame, variable=self.temperature_var, from_=0.0, to=1.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=150
        )
        self.temperature_slider.grid(row=0, column=3, padx=5)

        # System prompt
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

        self.manage_prompts_button = tk.Button(
            controls_frame, text="Manage Prompts", command=self.manage_prompts
        )
        self.manage_prompts_button.grid(row=2, column=3, pady=5)

        # ==== Chat area ====
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", height=20, width=80
        )
        self.chat_area.pack(padx=10, pady=10)

        # ==== Entry field ====
        entry_frame = tk.Frame(root)
        entry_frame.pack(pady=(0, 10))

        self.entry = tk.Entry(entry_frame, width=70)
        self.entry.pack(side=tk.LEFT, padx=(10, 0))
        self.entry.bind("<Return>", self.on_send)

        self.send_button = tk.Button(entry_frame, text="Send", command=self.on_send)
        self.send_button.pack(side=tk.LEFT, padx=10)

        self.save_chat_button = tk.Button(entry_frame, text="Save Chat", command=self.save_chat)
        self.save_chat_button.pack(side=tk.LEFT, padx=5)

        self.load_chat_button = tk.Button(entry_frame, text="Load Chat", command=self.load_chat)
        self.load_chat_button.pack(side=tk.LEFT, padx=5)

        self.messages = []
        self.response_queue = queue.Queue()
        self.poll_response_queue()
        self.load_prompts()

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

    def save_prompts(self):
        prompts_to_save = {k: v for k, v in self.system_prompts.items() if k != "Custom"}
        with open(self.PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts_to_save, f, indent=4)

    def reset_prompts(self):
        if os.path.exists(self.PROMPTS_FILE):
            os.remove(self.PROMPTS_FILE)
        self.system_prompts = self.DEFAULT_PROMPTS.copy()
        self.update_prompt_menu()
        messagebox.showinfo("Reset", "System prompts reset to defaults.")

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

    def manage_prompts(self):
        win = tk.Toplevel(self.root)
        win.title("Manage Prompts")

        listbox = tk.Listbox(win, height=10, width=40)
        listbox.pack(padx=10, pady=10)
        for name in self.system_prompts.keys():
            listbox.insert(tk.END, name)

        def delete_prompt():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel)
            if name in self.DEFAULT_PROMPTS:
                messagebox.showwarning("Cannot delete", f"'{name}' is a default prompt.")
                return
            del self.system_prompts[name]
            self.update_prompt_menu()
            self.save_prompts()
            listbox.delete(sel)

        def import_prompts():
            path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if not path:
                return
            with open(path, "r") as f:
                self.system_prompts.update(json.load(f))
            self.update_prompt_menu()
            self.save_prompts()
            win.destroy()

        def export_prompts():
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if not path:
                return
            with open(path, "w") as f:
                json.dump(self.system_prompts, f, indent=4)

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Delete Selected", command=delete_prompt).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Import", command=import_prompts).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export", command=export_prompts).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reset to Defaults", command=self.reset_prompts).pack(side=tk.LEFT, padx=5)

    # ==== Chat history ====

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


if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaChatApp(root)
    root.mainloop()
