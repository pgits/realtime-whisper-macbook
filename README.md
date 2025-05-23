This project runs with Python 3.  
I have also included the removal of the FutureWarning, here is what you can do if an updated library becomes available...
Update transformers: In the future, if you update your transformers library (e.g., pip install --upgrade transformers), this warning might disappear as newer versions of the pipeline fully adopt the input_features naming convention.

Prerequisites
Before you start, make sure you have the following installed on your MacBook:

Homebrew: A package manager for macOS. If you don't have it, install it by running this in your Terminal:
Bash

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Python 3.10 or newer:
Bash

brew install python
Verify your Python version:
Bash

python3 --version
FFmpeg: Essential for handling various audio formats.
Bash

brew install ffmpeg
Verify FFmpeg installation:
Bash

ffmpeg -version
pip (Python package installer):
Bash

pip3 install --upgrade pip
Step-by-Step Guide to Build the Program
Step 1.  Create a new virtual environment:

Bash
python3 -m venv venv

Step 2. Activate the new virtual environment:
Bash
source venv/bin/activate
Install PyTorch correctly:
This is the crucial step. You need to get the command from the official PyTorch website.

Go to https://pytorch.org/get-started/locally/
Select:
PyTorch Build: Stable
Your OS: macOS
Package: Pip
Compute Platform:
For Intel Macs: CPU
For Apple Silicon Macs (M1/M2/M3): MPS (This utilizes the GPU)
Copy the pip3 install command it generates. It will look something like this (for MPS):
Bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # (for Intel/CPU)
or (for Apple Silicon/MPS)
Bash

pip3 install torch torchvision torchaudio
Important Note for Apple Silicon/MPS: As of recent PyTorch versions, torch torchvision torchaudio alone (without --index-url) will often correctly detect and install the MPS-enabled version on Apple Silicon. The --index-url for cpu is typically for forcing CPU on Intel or for specific scenarios. Just use the command from the PyTorch website for your specific setup.
Install other required libraries:

Bash

pip install transformers accelerate sounddevice SpeechRecognition scipy
(Note: accelerate is beneficial, especially for MPS devices, but not strictly required for a basic setup. sounddevice and SpeechRecognition are for audio input.)

Verify PyTorch installation (within your active venv):

Bash

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
You should see True for MPS available if you have an Apple Silicon Mac and installed the MPS version.

Run your script:

Bash

python realtime_whisper.py
