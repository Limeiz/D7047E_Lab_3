import torch
from utils import print_examples
import torchvision.models as models

UNIQUE_TAG = "20240520-215050"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for stage in ['a', 'b', 'c']:
        print(f"Stage: {stage}")
        best = torch.load(f"best_model_{UNIQUE_TAG}{stage}.pt")
        print_examples(best['model'],
                       device,
                       transform=best['transform'],
                       vocab=best['vocab'],
                       )
