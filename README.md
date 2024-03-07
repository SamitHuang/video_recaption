# video_recaption

## Intallation

Install torch and run

```
pip install -r requirements.txt
```

## Usage

1. Download the coca model checkpoint from https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/tree/main

2. Put it under `models/coca/open_clip_pytorch_model.bin`

3. Caption video by

```
python recaption.py --video_path moscow.mp4
```
