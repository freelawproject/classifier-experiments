# Docketbert

The goal is to pretrain a domain-adapted language model for downstream classification and extraction tasks with docket entries.

## Pretraining Experiments

We conducted several small-scale pretraining experiments to evaluate different pretraining strategies. All experiments consist of a single epoch of training on 6.5M docket entries (~440M tokens). The models were trained on a single Nvidia A5000 24GB GPU.

For the detailed experimental setup, see the [`train.py`](train.py) script. To see the full tensorboard logs from these experiments, run the following command:

```bash
tensorboard --logdir home/projects/docketbert/runs
```

> **Note**: We also conducted an initial experiment using our `scratch-27M` config across 5 epochs. This model ultimately achieved a loss of 0.83, much lower than what is reported in the single-epoch results below. Because of this, and given that the loss curves for all experiments did not plateau, our final training runs should incorporate much more data and be trained for much longer to fully converge.

### Strategies

Here are some of the strategies we evaluated:

- **Baseline**: As a baseline, we fine-tuned modernbert-base and modernbert-large using a standard masked language modeling (MLM) objective.
- **Scratch Models**: We also trained several modernbert variants from scratch. For these we experimented with different architectural configurations such as different hidden sizes, number of layers, number of attention heads, and intermediate sizes.
- **Distillation**: We experimented with logit distillation from a teacher model. Here the training objective is a weighted combination of the standard MLM loss and the KL divergence between the student and teacher logits. See our [`DistillMLMRun`](../../clx/ml/distill_mlm_run.py) class for more details.
- **Sliced Models**: We also experimented with creating "sliced" models by reducing the depth of already-trained models and initializing the new layers with a subset of the weights from the original model. We compared approaches such as taking the first N layers versus taking every Nth layer.

### Inference

We conducted tests to evaluate model throughput in an inference setting.

- **GPU Tests**: We tested how fast 100k examples could be processed by a model on a single 24GB VRAM GPU. We autocasted layers to bfloat16, used flash attention, and did padding at the batch level. Each model was tested with the following batch sizes: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096. The speeds reported below are based on the model's best speed, ignoring any OOM errors.
- **CPU Tests**: We tested how fast 200 examples could be processed by a model on a single CPU. Models were tested on a device with 48GB RAM and 9 CPU cores. Models were tested with a batch size of 32.

### Results

|    | model                                                   |   100k_gpu_seconds |   200_cpu_seconds |   num_params_million |   eval_loss |
|---:|:--------------------------------------------------------|-------------------:|------------------:|---------------------:|------------:|
| 19 | docketbert-large-395M                                   |             81.054 |           175.448 |                  395 |    0.735635 |
| 20 | docketbert-large-(lr:2e-4)-395M                         |             81.07  |           178.15  |                  395 |    0.751795 |
| 14 | docketbert-sliced-large-ft-interleaved-10l-175M         |             40.64  |            53.647 |                  175 |    0.794689 |
|  8 | docketbert-distill-sliced-large-ft-interleaved-10l-175M |             38.226 |            54.568 |                  175 |    0.797992 |
| 18 | docketbert-base-150M                                    |             46.325 |            68.642 |                  150 |    0.814995 |
| 12 | docketbert-sliced-large-ft-interleaved-8l-150M          |             38.978 |            52.307 |                  150 |    0.839114 |
| 13 | docketbert-distill-sliced-large-ft-interleaved-8l-150M  |             39.848 |            46.144 |                  150 |    0.842217 |
| 17 | docketbert-sliced-large-interleaved-10l-175M            |             43.083 |            52.033 |                  175 |    0.864428 |
| 10 | docketbert-distill-sliced-large-interleaved-10l-175M    |             38.882 |            53.75  |                  175 |    0.869478 |
|  6 | docketbert-sliced-large-ft-interleaved-6l-126M          |             37.338 |            36.69  |                  126 |    0.895777 |
|  7 | docketbert-sliced-large-ft-first-6l-126M                |             37.472 |            31.121 |                  126 |    0.927982 |
|  4 | docketbert-sliced-large-interleaved-6l-126M             |             37.022 |            33.435 |                  126 |    0.942992 |
|  5 | docketbert-sliced-large-first-6l-126M                   |             37.098 |            32.831 |                  126 |    0.945664 |
|  9 | docketbert-distill-base-41M                             |             38.541 |            15.245 |                   41 |    1.11082  |
| 11 | docketbert-scratch-41M                                  |             38.953 |            15.113 |                   41 |    1.1354   |
|  1 | docketbert-sliced-base-interleaved-4l-59M               |             35.057 |            12.987 |                   59 |    1.14059  |
|  0 | docketbert-sliced-base-first-4l-59M                     |             34.656 |            14.506 |                   59 |    1.14098  |
| 15 | docketbert-distill-base-27M                             |             41.383 |            16.835 |                   27 |    1.18661  |
| 16 | docketbert-scratch-27M                                  |             42.565 |            16.45  |                   27 |    1.19237  |
|  3 | docketbert-scratch-16M                                  |             36.296 |             4.449 |                   16 |    1.3754   |
|  2 | docketbert-scratch-7M                                   |             35.633 |             3.177 |                    7 |    1.73775  |
| 21 | microsoft/deberta-v3-large                              |            481.898 |           171.154 |                  nan |  nan        |

### Takeaways

Here are some of the key takeaways from our experiments:

1. **Small models not significantly faster on GPU**: There seems to be a inference speed floor where even tiny models show little gain (<20%) over the `docketbert-base-150M` baseline. It is probably not worth trying to train a model from scratch given the diminished performance relative to the modest speedup.
2. **Distillation did not improve performance**: The distillation experiments achieved losses on-par with their non-distilled counterparts. Given that distillation makes training more costly, we will not pursue this approach further.
3. **CPU inference is more sensitive to model size, but much slower**: Inference on CPU does not seem to have the floor effect seen on GPU. Our tiny `docketbert-scratch-7M` model is more than 20x faster than the `docketbert-base-150M` baseline. That said, CPU is still massively slower than GPU, probably not adequate for production use.
4. **Sliced models with interleaved layers are better than first laters**: Across the board, taking every N layer from a larger model works better than taking the first N layers.
5. **Sliced models are good compromises**: Our sliced models based on a "large" architecture with 10 layers (such as `docketbert-sliced-large-ft-interleaved-10l-175M`) performed on-par with the `docketbert-base-150M` baseline. In spite of having slightly more parameters, they are nonetheless a little faster than the base model due to their reduced depth. Furthermore, although not reported below, they train nearly 2x faster than the base model.

### Next Steps

Based on the results, I think we should develop three variants of our docketbert model. These will include the two baseline variants based on modernbert-base and modernbert-large, as well as a 10 layer sliced model based on our finetune of modernbert-large. These correspond to the configurations from the following experiments:

- `docketbert-large-395M`
- `docketbert-base-150M`
- `docketbert-sliced-large-ft-interleaved-10l-175M`

We will train these on a larger dataset and for longer until convergence.
