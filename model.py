import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


class CornerPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(CornerPointNet, self).__init__()

    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    det_h = 65
    gn = 64
    useGn = False
    self.reBn = True
    if self.reBn:
        print ("model structure: relu - bn - conv")
    else:
        print ("model structure: bn - relu - conv")


    if useGn:
        print ("apply group norm!")
    else:
        print ("apply batch norm!")


    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)

    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
    self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
    self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2d(65)


  def forward(self, x, subpixel=False):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """

    # Let's stick to this version: first BN, then relu
    if self.reBn:
      # Shared Encoder.
      x = self.relu(self.bn1a(self.conv1a(x)))
      conv1 = self.relu(self.bn1b(self.conv1b(x)))
      x, ind1 = self.pool(conv1)
      x = self.relu(self.bn2a(self.conv2a(x)))
      conv2 = self.relu(self.bn2b(self.conv2b(x)))
      x, ind2 = self.pool(conv2)
      x = self.relu(self.bn3a(self.conv3a(x)))
      conv3 = self.relu(self.bn3b(self.conv3b(x)))
      x, ind3 = self.pool(conv3)
      x = self.relu(self.bn4a(self.conv4a(x)))
      x = self.relu(self.bn4b(self.conv4b(x)))
      # Detector Head.
      cPa = self.relu(self.bnPa(self.convPa(x)))
      semi = self.bnPb(self.convPb(cPa))
      # Descriptor Head.
      cDa = self.relu(self.bnDa(self.convDa(x)))
      desc = self.bnDb(self.convDb(cDa))
    else:
      # Shared Encoder.
      x = self.bn1a(self.relu(self.conv1a(x)))
      x = self.bn1b(self.relu(self.conv1b(x)))
      x = self.pool(x)
      x = self.bn2a(self.relu(self.conv2a(x)))
      x = self.bn2b(self.relu(self.conv2b(x)))
      x = self.pool(x)
      x = self.bn3a(self.relu(self.conv3a(x)))
      x = self.bn3b(self.relu(self.conv3b(x)))
      x = self.pool(x)
      x = self.bn4a(self.relu(self.conv4a(x)))
      x = self.bn4b(self.relu(self.conv4b(x)))
      # Detector Head.
      cPa = self.bnPa(self.relu(self.convPa(x)))
      semi = self.bnPb(self.convPb(cPa))

    dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
    output = {'semi': semi}
