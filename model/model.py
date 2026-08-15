import sys
sys.path.append('mednext')
from nnunet_mednext import create_mednext_v1
import torch

model = create_mednext_v1(
  num_input_channels = 3,
  num_classes = 1,
  model_id = 'S',             # S, B, M and L are valid model ids
  kernel_size = 3,            # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = False     # was used in publication
).cuda()

a = torch.randn(1, 3, 8, 160, 160).cuda()


# from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
