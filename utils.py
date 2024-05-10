import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, val_dataset):
    transform = val_dataset.transform

    model.eval()
    for index, (filename, caption) in enumerate([
        ("test_examples/dog.jpg", "Dog on a beach by the ocean"),
        ("test_examples/child.jpg", "Child holding red frisbee outdoors"),
        ("test_examples/bus.png", "Bus driving by parked cars"),
        ("test_examples/boat.png", "A small boat in the ocean"),
        ("test_examples/horse.png", "A cowboy riding a horse in the desert"),
        ("test_examples/1.jpg", "Man in black hat"),
    ]):
        test_img = transform(Image.open(filename).convert("RGB")).unsqueeze(0)
        print(f"Example {index} CORRECT: {caption}")
        print("           OUTPUT: "
              + " ".join(model.caption_image(test_img.to(device), val_dataset.dataset.vocab)))


    # # test_img7 = transform(
    #     Image.open("test_examples/2.jpg").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 6: Abdulkader and Santa")
    # print(
    #     "Example 6 OUTPUT: "
    #     + " ".join(model.caption_image(test_img7.to(device), dataset.vocab))
    # )
    # test_img8 = transform(
    #     Image.open("test_examples/3.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 6: Car")
    # print(
    #     "Example 6 OUTPUT: "
    #     + " ".join(model.caption_image(test_img8.to(device), dataset.vocab))
    # )
    # test_img9 = transform(
    #     Image.open("test_examples/4.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 6: Umbrellas")
    # print(
    #     "Example 6 OUTPUT: "
    #     + " ".join(model.caption_image(test_img9.to(device), dataset.vocab))
    # )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step