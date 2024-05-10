import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
                                        num_workers=2,
                                        )
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
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/" + use_data + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
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
        except FileNotFoundError as e:
            writer.add_text('except', "Checkpoint " + str(e), step)

    model.train()

    for epoch in range(num_epochs):
        writer.add_scalar('epoch', epoch, step)

        # Uncomment for test cases
        print_examples(model, device, dataset.get_loader(split="val"))

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
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()