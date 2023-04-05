# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode
        assert self.mode in ['C', 'S', 'M', 'L']

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'C':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'S':
            nr = 0
            for x in xs:
                nr += x
            return nr
        elif self.mode == 'M':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'L':
            return xs[-1]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
