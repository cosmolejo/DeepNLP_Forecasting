## Quick Start

- Prepare your dataset in the required format (see [Chronos TS documentation](https://github.com/amazon-science/chronos-forecasting)).
- Edit the config files in `scripts/training/configs/` as needed.
- Start fine-tuning:
    ```sh
    python scripts/training/train.py --config scripts/training/configs/chronos-t5-small-lora.yaml
    ```

## Fine-Tuning with LoRA

- LoRA adapters are saved in the `output/` directory after training.
- You can run batch experiments using the provided `.bat` or `.sh` scripts.

## Evaluation

- Evaluate checkpoints using:
    ```sh
    python scripts/evaluation/evaluate.py <config> <output_csv> --lora-model-id <checkpoint_dir> --batch-size 32 --device cuda:0 --num-samples 20
    ```
- Aggregate results with:
    ```sh
    python scripts/evaluation/agg-relative-score.py <checkpoint_name> --results-dir scripts/evaluation/results_max_step_exp
    ```

## Editable Mode for Development

If you want to modify the source code and see changes immediately, install in editable mode:
```sh
pip install --editable ".[training]"
```
Any changes to the code will be reflected without reinstalling.


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