@echo off
for /l %%N in (50,50,1000) do (
    python evaluation\evaluate.py ^
        evaluation\configs\fine-tuned.yaml ^
        evaluation\results_max_step_exp\chronos-t5-small-lora-cp%%N-fine-tuned.csv ^
        --lora-model-id "output\run-0\checkpoint-%%N" ^
        --batch-size=32 ^
        --device=cuda:0 ^
        --num-samples 20

    python evaluation\agg-relative-score.py "chronos-t5-small-lora-cp%%N" ^
        --is-fine-tuned-required ^
        --no-is-in-domain-required ^
        --no-is-zero-shot-required ^
        --results-dir .\evaluation\results_max_step_exp
)