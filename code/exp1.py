# %%
# PyTorch
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

# For dislaying images
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Loading dataset
from datasets import load_dataset

# Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

# Matrix operations
import numpy as np

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
from dotenv import load_dotenv

ENV_FILE = "/home/rnanawa1/GeoLifeLearn/.env"
load_dotenv(ENV_FILE)

# %%
from datasets import load_dataset, Image

DATA_DIR = "/home/rnanawa1/GeoLifeLearn/data/species25/"
dataset = load_dataset(DATA_DIR).cast_column("image", Image(decode=True))

# %%
trainds = dataset["train"]
valds = dataset["validation"]
testds = dataset["test"]

# %%
trainds.features["label"]

# %%
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)

mu, sigma = processor.image_mean, processor.image_std  # get default mu,sigma
size = processor.size

# %%
norm = Normalize(mean=mu, std=sigma)  # normalize image pixels range to [-1,1]

# resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
_transf = Compose([Resize(size["height"]), ToTensor(), norm])


# apply transforms to PIL Image and store it to 'pixels' key
def transf(arg):
    arg["pixels"] = [_transf(image.convert("RGB")) for image in arg["image"]]
    return arg


# %%
trainds.set_transform(transf)
valds.set_transform(transf)
testds.set_transform(transf)

# %%
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=25, ignore_mismatched_sizes=True
)
print(model.classifier)

# %%
args = TrainingArguments(
    "ViT_Exp1",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="logs",
    remove_unused_columns=False,
)

print(f"Args Device:{args.device}")


def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


trainer = Trainer(
    model,
    args,
    train_dataset=trainds,
    eval_dataset=valds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


trainer.train()

outputs = trainer.predict(testds)
print(outputs.metrics)

# %%
