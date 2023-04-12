## Data
Download the librispeech [data](https://drive.google.com/file/d/1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE/view) from https://github.com/CorentinJ/librispeech-alignments.
Then unzip the data into the directory Librispeech. The directory should look like 
```
speech_diff/
    Librispeech/
        dev-clean/
        dev-other/
        ...
```
Then run the following command to extract the texts:
```sh
mkdir datasets
python tools/extract_sentence.py
```

## Train the model (optional)
```sh
bash scripts/train.sh
```
Trained model url: https://drive.google.com/file/d/1AcRed0QuGe3b54eexaWSlgBV7eCngPH7/view?usp=sharing

Download it to your server and change the corresponding cmd argument before runing generation or evaluation (--encoder_path)

The training process can be found at https://wandb.ai/bairu/pl_diffusion/runs/8dvnjfeu/


## Generate (Reconstruction)

```sh
bash scripts/generate.sh
```

## Evaluate Reconstruction WER

```sh
bash scripts/eval_wer.sh
```
WER of the trained model: 11.1%

