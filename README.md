------------------------------------------

1. Install and use Ollama

1.1.Download and install:
curl -fsSL https://ollama.com/install.sh | sh

Check installation:
ollama --version

1.2. Configure the service:
Update this file: /etc/systemd/system/ollama.service with this content (adjust it):
'''
[Unit]
Description=Ollama AI Service
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
User=caico
Group=caico
WorkingDirectory=/home/caico
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
'''

1.3. Commands for managing the service:
sudo systemctl start ollama	Start it manually
sudo systemctl stop ollama	Stop it
sudo systemctl restart ollama	Restart it
systemctl is-enabled ollama	Check if enabled at boot
sudo systemctl enable ollama	(optional) Enable auto-start at boot
sudo systemctl disable ollama	Prevent auto-start at boot

1.4. After starting the service, pull a model:
ollama pull llama3.1:8b

1.5. When starting service and pulling a model, chat with it:
ollama run llama3.1:8b

1.6. Stop chatting by typping:
/bye
Or press Ctrl-D or Ctrl-C (not recommended)



2. Basic Python script for using Ollama

2.1. Createa a new directory and a python virtual environment on it:
python3 -m venv venv

2.2. Activate virtual env:
source venv/bin/activate

2.3. Install ollama for python:
pip install ollama

2.4. Create a requirements.txt with this content:
'''
ollama>=0.1.7
black>=24.0.0      # for code formatting
flake8>=7.0.0      # for linting
pytest>=8.0.0      # for testing
'''

2.5. Install the requirements:
pip install -r requirements.txt

2.6. Create the ollama hello world python scrpt, run the ollama model (1.5) and run the script it:

