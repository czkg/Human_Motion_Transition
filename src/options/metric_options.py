#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from .base_options import BaseOptions


class MetricOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--current_s', required=True, help='which sub to select')
        #parser.add_argument('--n_steps', type=int, help='steps of the integral')
        parser.add_argument('--n_neighbors', type=int, help='number of neighbors of each z')
        parser.add_argument('--z0', required=True, help='the start point of the path')
        parser.add_argument('--z1', required=True, help='the end point of the path')
        parser.add_argument('--store_path', type=str, help='path to store the graph')
        self.isTrain = False
        return parser
