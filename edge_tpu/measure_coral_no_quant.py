# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
import time

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import numpy as np

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')

  args = parser.parse_args()

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  path2data = "testseqs.npz"

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
  test = testData.astype(np.float32)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  output_data = np.zeros_like(test)
  inference_time = []
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for i in range(100):
    # set tensor
    t = np.expand_dims(test[i], axis=0).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]['index'], t)
    #perform inference
    start = time.perf_counter()
    interpreter.invoke()
    inference_time.append(time.perf_counter() - start)
    output_data[i] = interpreter.get_tensor(output_details[0]['index'])

  tmp = np.array(inference_time[1:]) * 1000
  print('%.1fms' % np.mean(tmp))

  print('-------RESULTS--------')
  print(output_data[0])

  def rmse(a,b):
    tmp = (a-b)**2
    tmp = np.sum(tmp, axis=1)
    return np.sqrt(tmp)/np.shape(a)[1]

  print(rmse(test, output_data).mean())

if __name__ == '__main__':
  main()
