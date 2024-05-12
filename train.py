import datetime

UNIQUE_TAG = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import EISL
from get_loader import get_batch_loader, get_dataset
from model import CNNtoRNN
from utils import save_checkpoint, load_checkpoint, print_examples


def train():
    use_data = "coco"

    if use_data == "flickr":
        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_transform = transforms.Compose(
            [
                transforms.Resize((356, 356)),
                transforms.RandomCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = get_dataset(
            root_folder="flickr8k/images",
            annotation_file="flickr8k/captions.txt",
            transformers={'train': train_transform,
                          'val': transform,
                          'test': transform,
                          },
        )
        train_loader = get_batch_loader(dataset,
                                        split="train",
                                        num_workers=1,
                                        )
        val_loader = get_batch_loader(dataset,
                                      split="val",
                                      num_workers=1,
                                      )
        val_loader.dataset.use_only_first_caption = False

    elif use_data == "coco":
        dataset = get_dataset(
            root_folder="data/coco",
            transform=models.Inception_V3_Weights.DEFAULT.transforms(),
            annotation_file="data/coco/dataset_coco.json",
        )
        train_loader = get_batch_loader(dataset,
                                        split='train',
                                        num_workers=4,
                                        )
        val_loader = get_batch_loader(dataset,
                                      split='val',
                                      num_workers=4,
                                      )
    else:
        raise NotImplementedError("Dataset not implemented: " + use_data)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 2048
    hidden_size = 2048
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1000

    # for tensorboard
    writer = SummaryWriter("runs/" + use_data + "/" + UNIQUE_TAG)
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    eisl_7030 = EISL.EISLNatCriterion(
        # task=,
        label_smoothing=0.0,
        ngram="3,5",
        ce_factor=0.7,
        ngram_factor=0.3,
        weights=dataset.vocab.weights.to(device),
    )
    eisl_3070 = EISL.EISLNatCriterion(
        # task=,
        label_smoothing=0.0,
        ngram="3,5",
        ce_factor=0.3,
        ngram_factor=0.7,
        weights=dataset.vocab.weights.to(device),
    )
    eisl_0100 = EISL.EISLNatCriterion(
        # task=,
        label_smoothing=0.0,
        ngram="3,4",
        ce_factor=0.0,
        ngram_factor=1.0,
        weights=None,
    )

    criterion = eisl_7030.compute_EISL
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        try:
            step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
            print("Loaded checkpoint, continuing at", step)
            writer.add_text('info', "Continue run", global_step=step)
        except FileNotFoundError as e:
            writer.add_text('except', "No Checkpoint " + str(e), global_step=step)

    best_state_dict = model.state_dict()
    min_validation_loss = float("inf")
    no_improvement = 0
    stage = 'a'
    for epoch in range(num_epochs):
        writer.add_scalar('epoch', epoch, step)

        # Uncomment for test cases
        print_examples(model,
                       device,
                       transform=dataset.transformers['val'],
                       vocab=dataset.vocab)

        model.train()

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            losses = criterion(
                outputs.permute([1, 0, 2]),
                captions.permute([1, 0]),
            )
            loss = losses['loss'] if isinstance(losses, dict) else losses

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_sentences = 0
        for index, (imgs, captions) in enumerate(val_loader):
            nr_sentences = captions.shape[1]
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            losses = criterion(
                outputs.permute([1, 0, 2]),
                captions.permute([1, 0]),
            )
            if isinstance(losses, dict):
                writer.add_scalars("validation batch losses", {k: (v.item() if isinstance(v, torch.Tensor) else v)
                                                    for k, v in losses.items() if 'loss' in k},
                                   global_step=step + index)
            total_loss += nr_sentences * (losses['loss'].item() if isinstance(losses, dict) else losses.item())
            total_sentences += nr_sentences
        average_loss = total_loss / total_sentences
        writer.add_scalar("Validation loss", average_loss, global_step=step)
        if average_loss < min_validation_loss:
            min_validation_loss = average_loss
            no_improvement = 0
            print(f"=> Saving best model[{stage}], validation loss {min_validation_loss}")
            torch.save({'model': model,
                       'transform': dataset.transformers['val'],
                       'vocab': dataset.vocab,
                        },
                       f"best_model_{UNIQUE_TAG}{stage}.pt")
            best_state_dict = model.state_dict()
        else:
            no_improvement += 1
            if no_improvement % 10 == 0:
                min_validation_loss = float("inf")
                model.load_state_dict(best_state_dict)  # restart from best with new loss
                if criterion == eisl_7030.compute_EISL:
                    criterion = eisl_3070.compute_EISL
                    stage = 'b'
                    writer.add_text('info', "Switching to higher weight EISL", global_step=step)
                    print(f"=> Switched to higher weight EISL")
                elif criterion == eisl_3070.compute_EISL:
                    # fine tune with EISL only
                    criterion = eisl_0100.compute_EISL
                    stage = 'c'
                    writer.add_text('info', "Switching to fine tune with EISL only", global_step=step)
                    print(f"=> Switched to fine tune with EISL only")
                elif criterion == eisl_0100.compute_EISL:
                    break
        writer.add_scalar("No improvements count", no_improvement, global_step=step)


if __name__ == "__main__":
    train()
