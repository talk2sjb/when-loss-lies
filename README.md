### When Loss Lies
>_Learning arithmetic rules long before correctness, and why grokking didn’t happen (yet).
Investigating memorization, generalization, and grokking in arithmetic transformers_

This is purely research based study in my pursuit of understanding how models learn through experimentation.

Detailed publication can be found in my personal [blog](https://wsdmbox.com/2025/12/22/why-arithmetic-models-look-dumb-long-after-theyve-learned-the-rule/)

#### To run training the model
```sh
python train_math_llm.py
```
#### To resume training a model from a checkpoint
```sh
python train_math_llm.py --epochs 750 --batch_size 30 --output_dir math_model_output_750_epoch --checkpoint ./math_model_output_500_epoch/checkpoint-150000
```
