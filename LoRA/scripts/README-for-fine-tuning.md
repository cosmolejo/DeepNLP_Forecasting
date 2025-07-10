## Chronos & LoRA: Learning fine details efficiently :)
Before diving into fine-tuning, let's get your environment ready with the following steps:
1. **Create a Python virtual environment:**
    <pre>
    `python -m venv [name of the environment]`
    </pre>

2. Activate the virtual environment:
    <pre>
    `.\[name of the environment]\Scripts\activate`
    </pre>
3. Open a terminal inside the LoRA folder (located within the repository), and install Chronos in editable mode with training dependencies:
    <pre>
    `pip install --editable ".[training]"`
    </pre>
4. **Install PEFT (Parameter-Efficient Fine-Tuning)** to enable LoRA support: 
    <pre>
    `pip install peft`
    </pre>
5. **Install the CUDA-enabled version of PyTorch** to leverage GPU acceleration (if available): 
    - If you are on **Windows 11** with an NVIDIA GPU: 
    <pre>
        `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
    </pre>
    - On other platforms, please visit the [official PyTorch install guide](https://pytorch.org/get-started/locally/) to find the right command for your system
6. **Start fine-tuning:** Navigate to the scripts folder and run:
`python training/train.py --config ./training/configs/chronos-t5-small-lora.yaml`.
    <pre>
   This command uses the configuration specified in: `./training/configs/chronos-t5-small-lora.yaml`
    </pre>