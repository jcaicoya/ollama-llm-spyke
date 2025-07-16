
# Table of contents
1. [Install Ollama on Linux](#1-install-ollama)
    1. [Download and install](#11-download-and-install)
    2. [Configure the service](#12-configure-the-service)
    3. [Commands for managing the service](#13-commands-for-managing-the-service)
    4. [Pull a model](#14-pull-a-model)
2. [Run and stop Ollama](#2-run-and-stop-ollama)
    1. [Run a model](#21-run-a-model)
    2. [Stop running](#22-stop-running)
3. [Python virtual environment](#3-create-and-run-a-python-virtual-enviroment)
    1. [Create a python venv](#31-create-a-python-venv)
    2. [Activate venv](#32-activate-virtual-env)
    3. [Create requirements.txt](#33-create-a-requirementstxt)
    4. [Intall the requirements](#34-install-the-requirements)
    5. [Deactivate venv](#35-deactivate-virtual-env)
4. [Ollama checker pythng scirpt](#4-ollama-checker-python-script)
    1. [Create a basic checker python script](#41-create-a-basic-checker-python-script)
    2. [Run the basic script](#42-run-the-basic-script)

## 1. Install Ollama

### 1.1. Download and install

    curl -fsSL https://ollama.com/install.sh | sh

Check installation:

    ollama --version

### 1.2. Configure the service:

Update this file: **/etc/systemd/system/ollama.service** with this content (adjust it):

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

### 1.3. Commands for managing the service:

- Start it:                     sudo systemctl start ollama	
- Stop it:                      sudo systemctl stop ollama	
- Restart it:                   sudo systemctl restart ollama	
- Check if enabled at boot:     systemctl is-enabled ollama
- Enable auto-start at boot:    sudo systemctl enable ollama
- Disable auto-start at boot:   sudo systemctl disable ollama

### 1.4. Pull a model

List of models: [Ollama models](https://ollama.com/library))
As example, we are going to pull llama3.1.:8b model:

    ollama pull llama3.1:8b


## 2. Run and stop Ollama

### 2.1. Run a model

We assume that Ollama service is up and listening; then launch this command

    ollama run llama3.1:8b

We can start chatting with the model

### 2.2. Stop running

Type **/bye** or press **Ctrl-D**


## 3. Create and run a Python virtual enviroment

### 3.1. Create a Python venv:

    python3 -m venv venv

### 3.2. Activate virtual env

    source venv/bin/activate

### 3.3. Create a requirements.txt

Create this file in proyect root directory with this content:

    ollama>=0.1.7
    black>=24.0.0      # for code formatting
    flake8>=7.0.0      # for linting
    pytest>=8.0.0      # for testing

### 3.4. Install the requirements:

    pip install -r requirements.txt

### 3.5. Deactivate virtual env

    deactivate


## 4. Ollama checker python script

### 4.1. Create a basic checker python script

The basic script could be this one:

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

### 4.2. Run the basic script

Make sure that:
- Ollama service is up and listening
- We are in the python venv

Then launch:

    python3 basic-ollama-checker.py
