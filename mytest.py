import numpy as np
import torch
from torchvision.transforms import functional as F
import math, random
from PIL import Image



print(np.__version__)
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(torch.version.cuda)