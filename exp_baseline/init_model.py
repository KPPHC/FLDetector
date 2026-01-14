#!/usr/bin/env python3
import torch
import torchvision.models as models


def main() -> None:
    torch.manual_seed(42)
    model = models.resnet18(weights=None)
    torch.save(model.state_dict(), "resnet18_init.pth")
    print("Saved resnet18_init.pth (seed=42)")


if __name__ == "__main__":
    main()
