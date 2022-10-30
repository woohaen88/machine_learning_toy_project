from neural_lib import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


def preprocess_image(image: str):
    # BGR -> RGB로 변환
    # resize
    # value range [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    # 1. 표준화
    # 2. 채널을 처음으로, 배치 dimension 추가
    image -= config.MEAN
    image /= config.STD
    image = np.transpose(
        image, (2, 0, 1)
    )  # 현재 이미지(width, height, channel) -> (channel, width, height)
    image = np.expand_dims(image, 0)  # (1, channel, width, height)

    return image


# argument parser와 arguments parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path th the input image")
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="vgg16",
    choices=["vgg16", "vgg19", "inception", "densenet", "resnet"],
    help="name of pre-trained network to use",
)
args = vars(ap.parse_args())

# 모델이름과 클래스를 dictionary로 설정
MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "inception": models.inception_v3(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
    "resnet": models.resnet50(pretrained=True),
}

# 가중치를 불러오고 메모리에 저장
print("[INFO] loading {}...".format(args["model"]))  # vgg16, vgg19 ...
model = MODELS[args["model"]].to(config.DEVICE)  # config.DEVICE := cpu
model.eval()


# 이미지를 불러옴
print("[INFO] loading image ... ")
image = cv2.imread(args["image"])
orig = image.copy()  # image 복사
image = preprocess_image(image)

# 현재 이미지 타입이 numpy.array이므로 torch.tensor로 변환해주어야함
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# 전처리된 이미지 라벨
print("[INFO] loading ImageNet labels...")
imagenet_labels = dict(enumerate(open(config.IN_LABELS)))

# 이미지 분류하고 예측을 뽑아냄
print("[INFO] classifying image with '{}'...".format(args["model"]))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sorted_proba = torch.argsort(probabilities, dim=-1, descending=True)

# 최대 5개까지 출력
for (i, idx) in enumerate(sorted_proba[0, :5]):
    print(
        "{}. {}: {:.2f}%".format(
            i,
            imagenet_labels[idx.item()].strip(),
            probabilities[0, idx.item()] * 100,
        )
    )

(label, prob) = (
    imagenet_labels[probabilities.argmax().item()],
    probabilities.max().item(),
)
cv2.putText(
    orig,
    "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2,
)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
