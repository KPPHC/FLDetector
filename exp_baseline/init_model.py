python3 - << 'PY'
import torch
import torchvision.models as models

m = models.resnet18(weights=None)   # random init
torch.save(m.state_dict(), "resnet18_init.pth")
print("saved resnet18_init.pth")
PY
