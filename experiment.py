import wandb
import torch
import argparse
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset, Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.transforms import Normalize, Resize, ToTensor, Compose


def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}


def compute_metrics(eval_pred, criterion=torch.nn.CrossEntropyLoss()):
    predictions, labels = eval_pred

    bce_loss = criterion(torch.tensor(predictions), torch.tensor(labels))
    return dict(
        loss=bce_loss, accuracy=accuracy_score(np.argmax(predictions, axis=1), labels)
    )


def main(args):

    dataset = load_dataset("nrishabh/geolife", cache_dir=args.data_dir)

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
        try:
            arg["pixels"] = [_transf(image.convert("RGB")) for image in arg["image"]]
            return arg
        except KeyError:
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
    probs = outputs.predictions
    y_pred = outputs.predictions.argmax(1)

    wandb.log(
        {
            "conf_mat_probs": wandb.plot.confusion_matrix(
                probs=probs, y_true=y_true, class_names=range(1, 26)
            )
        }
    )
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=range(1, 26)
            )
        }
    )

    wandb.log({"roc": wandb.plots.ROC(y_true, probs)})
    wandb.log({"pr": wandb.plots.precision_recall(y_true, probs)})

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred
    )

    wandb.log({"avg_precision": np.mean(precision)})
    wandb.log({"avg_recall": np.mean(recall)})
    wandb.log({"avg_fscore": np.mean(fscore)})
    wandb.log({"avg_support": np.mean(support)})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GeoLifeLearn Experiment")
    parser.add_argument(
        "--env", type=str, default="./.env", help="Environment Variables file"
    )
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
        default="./data/",
        help="Directory path to the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    args = parser.parse_args()

    load_dotenv(args.env)

    main(args)
