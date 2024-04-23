'''
Script to convert a trained CenterNet model to ONNX, currently only
support non-DCN models.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy

from lib.model.model import create_model, load_model
from lib.opts import opts
from lib.dataset.dataset_factory import dataset_factory
from lib.detector import Detector

from checker import checker
import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

def convert_onnx(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.model_output_list = True
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda')
  else:
    opt.device = torch.device('cpu')
  opt.device = torch.device('cpu')
  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
  if opt.load_model != '':
    model = load_model(model, opt.load_model, opt)
  model = model.to(opt.device)
  model.train(True)
  dummy_input1 = torch.randn(1, 3, opt.input_h, opt.input_w).to(opt.device)

  if opt.tracking:
    dummy_input2 = torch.randn(1, 3, opt.input_h, opt.input_w).to(opt.device)
    if opt.pre_hm:
      dummy_input3 = torch.randn(1, 1, opt.input_h, opt.input_w).to(opt.device)
      print('------------')
      print('------------')
      print('FIRST ONE')
      print('------------')
      print('------------')
      torch.onnx.export(
        model, (dummy_input1, dummy_input2, dummy_input3), 
        "../models/{}.onnx".format(opt.exp_id))
    else:
      print('------------')
      print('------------')
      print('SECOND ONE')
      print('------------')
      print('------------')
      torch.onnx.export(
        model, (dummy_input1, dummy_input2), 
        "../models/{}.onnx".format(opt.exp_id))
  else:
    print('------------')
    print('------------')
    print('THIRD ONE')
    print('------------')
    print('------------')
    path = '/home/hammad2002/Desktop/centerfusionpp/model.onnx'
    print(type(model))
    torch.onnx.export(
      model, 
      (dummy_input1, ), 
      path,
      input_names=['input'], 
      output_names=['outputOne', 'outputTwo', 'outputThree'], 
      opset_version=12, 
      verbose=True, 
      keep_initializers_as_inputs=False, 
      do_constant_folding=True, 
      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
      training=torch.onnx.TrainingMode.TRAINING, # default .EVAL is used
      export_params=True)
      # "../models/{}.onnx".format(opt.exp_id))
    model_outputs = model(dummy_input1)
    checker(path=path, model_outputs=model_outputs, input=dummy_input1)

if __name__ == '__main__':
  opt = opts().parse()
  convert_onnx(opt)