# DeepNLP_Forecasting: Chronos TS Adaptation with LoRA

This repository adapts [Chronos: Learning the Language of Time Series](https://openreview.net/forum?id=gerNCVqqtR) for efficient fine-tuning using LoRA (Low-Rank Adaptation) techniques.  
Chronos TS is developed by Amazon Science; this project extends its capabilities for research and experimentation.

## Table of Contents

- [About Chronos TS](#about-chronos-ts)
- [About This Adaptation](#about-this-adaptation)
- [Installation](#-installation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About Chronos TS

Chronos TS is a state-of-the-art framework for time series forecasting using language models.  
For details, see the [Chronos TS paper](https://openreview.net/forum?id=gerNCVqqtR) and [original repository](https://github.com/amazon-science/chronos-forecasting).

## About this Adaptation

This project adapts Chronos TS to support LoRA-based fine-tuning, enabling parameter-efficient transfer learning for time series models.  
Key changes include:
- Integration of [PEFT](https://github.com/huggingface/peft) for LoRA support
- Scripts for fine-tuning and evaluation on custom datasets

## 🚀 Installation
To install with all necessary dependencies, including evaluation tools and PEFT support:

1. **Clone this repository**
    ```sh
    git clone https://github.com/cosmolejo/DeepNLP_Forecasting
    cd DeepNLP_Forecasting/LoRA
    ```

2. **Create a Python virtual environment**
    ```sh
    conda create --name .venv
    conda activate .venv 
    ```

3. **Install in editable mode with training and evaluation dependencies**
    ```sh
    pip install --editable ".[training, evaluation]"
    ```

4. **Install PEFT and CUDA-enabled PyTorch**
    ```sh
    pip install peft
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
    > **Note:**  
    > The URL `https://download.pytorch.org/whl/cu128` is for CUDA 12.8.  
    > Make sure to use the correct URL for your CUDA version.  
    > See the [official PyTorch install guide](https://pytorch.org/get-started/locally/) for details.

Now that your environment is set up, you can find scripts for fine-tuning with LoRA and evaluating Chronos models in the [scripts](./scripts/) folder.


## Citation

If you use this adaptation, please cite Chronos TS:

```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}
```

## License

This adaptation follows the original Chronos TS license (Apache-2.0).

## Acknowledgements

- [Chronos TS](https://github.com/amazon-science/chronos-forecasting) by Amazon Science
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
