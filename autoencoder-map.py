from __future__ import absolute_import, print_function
import seaborn as sns

import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
for i, c in enumerate(colors):
    colors[i] = c[0:7] + "20"

dtype = torch.float

def makeLoader(inputs, batchSize=128):
  inputs = torch.tensor(inputs, dtype=dtype)

  return torch.utils.data.DataLoader(
      inputs,
      shuffle=True,
      batch_size=batchSize,
      num_workers=5)

def loadTensor(fileName):
  if fileName.endswith("npz"):
    return np.load(fileName, encoding="bytes", allow_pickle=True)["arr_0"]

inputSize = 3*1024

class GrindNet(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    sizes = [inputSize, 256]
    self.layers = nn.ModuleList()

    for size1, size2 in zip(sizes, sizes[1:]):
      self.layers.append(nn.Linear(in_features=size1, out_features=size2))
    for size1, size2 in zip(reversed(sizes), list(reversed(sizes))[1:]):
      self.layers.append(nn.Linear(in_features=size1, out_features=size2))
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
      x = functional.leaky_relu(x)
    return x

mode = "standard"
model = GrindNet()
print(model)

#testData = loadTensor("dba/testseqs.npz")

def flattenData(data):
  data = list(data)
  for i, seq in enumerate(data):
    seq = seq[1:]
    data[i] = seq.flatten()
  data = np.array(data)
  return data

#testData = flattenData(testData)

def preprocess(data):
  for seq in data:
    maximum = np.max(seq)
    minimum = np.min(seq)
    seq[:] -= minimum
    seq[:] /= maximum
  return data

stateName =  "net-state-3072-scaled2-{0}".format(mode)

model.load_state_dict(torch.load(stateName))

model = model.eval()

def evalModel(model, inputData):
  return model(torch.tensor(preprocess([inputData])[0], dtype=dtype)).detach().numpy()

#predictions = []
#losses = []
#differences = []
#with torch.no_grad():
#  for seq in testData:
#    output = evalModel(model, seq)
#    seq = preprocess([seq])[0]
#    differences.append(np.abs(seq - output))
#    loss = np.sqrt(np.square(seq - output).mean())
#    predictions.append(output)
#    losses.append(loss)

#  torch.onnx.export(model,
#      torch.randn(*(inputSize,)),
#      "model.onnx",
#      export_params=True,
#      opset_version=10,
#      do_constant_folding=True,
#      input_names=['input'],
#      output_names=['output'])
