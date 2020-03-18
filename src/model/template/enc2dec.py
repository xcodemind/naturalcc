# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.model.template.model_template import IModel


class Encoder2Decoder(IModel):
    def __init__(self, encoder: Module, decoder: Module, ):
        super(Encoder2Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __str__(self):
        return '{}(\n{}\n{}\n)'.format(self.__class__.__name__, self.encoder, self.decoder)
