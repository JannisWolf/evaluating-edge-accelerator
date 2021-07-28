#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
#import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt
import time
import timeit
import numpy as np

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.", required=True,
                      type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="MYRIAD", type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = args.model
    #log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    #net.batch_size = len(args.input)

    # Read and pre-process input images
    n = net.input_info[input_blob].input_data.shape
    #print("{}".format(n))
    with open('test.npy', 'rb') as f:
        images = np.load(f).astype(np.float16)

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    path2data="testseqs.npz"

    def loadTensor(fileName):
      if fileName.endswith("npz"):
        return np.load(fileName, encoding="bytes", allow_pickle=True)["arr_0"]

    def flattenData(data):
      data = list(data)
      for i, seq in enumerate(data):
        seq = seq[1:]
        data[i] = seq.flatten()
      data = np.array(data)
      return data

    def preprocess(data):
      for seq in data:
        maximum = np.max(seq)
        minimum = np.min(seq)
        seq[:] -= minimum
        seq[:] /= maximum
      return data

    testData = loadTensor(path2data)
    testData = flattenData(testData)
    testData = preprocess(testData)
    testData = testData.astype(np.float16)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    inference_time = []
    res = np.zeros_like(testData)
    for i in range(100):
      start = time.time()
      res[i] = exec_net.infer(inputs={input_blob: testData[i]})[out_blob]
      stop = time.time()
      inference_time.append(stop-start)

    # Processing output blob
    log.info("Processing output blob")
    t = np.array(inference_time) * 1000
    log.info("Time per sample: {:.2f}ms".format(np.sum(t)/100))
    def rmse(a,b):
      tmp = (a-b)**2
      tmp = np.sum(tmp, axis=1)
      return np.sqrt(tmp)/np.shape(a)[1]
    log.info("Average RMSE: {:.7f}".format(rmse(testData, res).mean()))

if __name__ == '__main__':
    sys.exit(main() or 0)
