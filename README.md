## Chronos & LoRA: Learning fine details efficiently :)
Before diving into fine-tuning, let's get your environment ready with the following steps:
1. **Create a Python virtual environment:**
    <pre>
    `python -m venv [name of the environment]`
    </pre>
2. Open a terminal inside the LoRA folder (located within the repository), and install Chronos in editable mode with training dependencies:
    <pre>
    `pip install --editable ".[training]"`
    </pre>
3. **Install PEFT (Parameter-Efficient Fine-Tuning)** to enable LoRA support: 
    <pre>
    `pip install peft`
    </pre>
4. **Install the CUDA-enabled version of PyTorch** to leverage GPU acceleration (if available): 
    - If you are on **Windows 11** with an NVIDIA GPU: 
    <pre>
        `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
    </pre>
    - On other platforms, please visit the [official PyTorch install guide](https://pytorch.org/get-started/locally/) to find the right command for your system
5. **Start fine-tuning:** Navigate to the scripts folder and run:
`python training/train.py --config ./training/configs/chronos-t5-small-lora.yaml`.
    <pre>
   This command uses the configuration specified in: `./training/configs/chronos-t5-small-lora.yaml`
    </pre>

```
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}
```