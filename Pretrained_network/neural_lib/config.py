from typing_extensions import Final

import torch

IMAGE_SIZE: Final = 224

# ImageNet의 평균과 표준편차
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# device 세팅
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet labels 경로
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"
