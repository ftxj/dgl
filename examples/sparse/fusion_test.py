
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_l = nn.Parameter(torch.zeros(1, 10, 20))
        self.a_r = nn.Parameter(torch.zeros(1, 10, 20))

    def forward(self, Z, row, col):
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[row] + e_r[col]


from torch._dynamo import config
config.suppress_errors = True

import logging
from torch._inductor import config
config.debug = True

from torch._dynamo import config as config2
config2.log_level = logging.DEBUG
config2.output_code = True

x = Net().to('cuda')

x = torch.compile(x)

x(torch.randn(1, 10, 20).to('cuda'), torch.randint(0, 1, [10]).to('cuda'), torch.randint(0, 1, [10]).to('cuda'))