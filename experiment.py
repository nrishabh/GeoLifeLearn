import wandb
import torch
import argparse
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset, Image
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}


def compute_metrics(eval_pred, criterion=torch.nn.CrossEntropyLoss()):
    predictions, labels = eval_pred

    ce_loss = criterion(torch.tensor(predictions), torch.tensor(labels))
    return dict(
        loss=ce_loss, accuracy=accuracy_score(np.argmax(predictions, axis=1), labels)
    )

def main(args):

    dataset = load_dataset("nrishabh/geolife", cache_dir="data/")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project="geolearn",
        name=f"{args.model_name}_{timestamp}",
        tags=[args.model_name],
    )

    dataset = load_dataset(args.data_dir).cast_column("image", Image(decode=True))

    trainds = dataset["train"]
    valds = dataset["validation"]
    testds = dataset["test"]

    processor = ViTImageProcessor.from_pretrained(args.model_name)

    mu, sigma = processor.image_mean, processor.image_std  # get default mu,sigma
    size = processor.size

    norm = Normalize(mean=mu, std=sigma)  # normalize image pixels range to [-1,1]

    # resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
    _transf = Compose([Resize(size["height"]), ToTensor(), norm])

    # apply transforms to PIL Image and store it to 'pixels' key
    def transf(arg):
        arg["pixels"] = [_transf(image.convert("RGB")) for image in arg["image"]]
        return arg

    trainds.set_transform(transf)
    valds.set_transform(transf)
    testds.set_transform(transf)

    model = ViTForImageClassification.from_pretrained(
        args.model_name, num_labels=25, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        f"{args.model_name}_{timestamp}",
        save_strategy="no",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"{args.model_name}_{timestamp}",
        logging_steps=500,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=trainds,
        eval_dataset=valds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()

    outputs = trainer.predict(testds)
    
    y_true = testds["label"]

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=outputs.predictions,
                            y_true=y_true)})

    wandb.log({'roc': wandb.plots.ROC(y_true, outputs.predictions)})
    wandb.log({'pr': wandb.plots.precision_recall(y_true, outputs.predictions)})




if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="GeoLifeLearn Experiment")
    parser.add_argument("dataset_directory", type=str, default="data/", help="Path to the dataset directory")
    parser.add_argument(
        "model_name", type=str, help="Name of the pre-trained ViT model"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/rnanawa1/GeoLifeLearn/data/species25/",
        help="Directory path to the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    args = parser.parse_args()

    main(args)