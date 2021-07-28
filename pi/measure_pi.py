import tensorflow as tf
import numpy as np
from tensorflow import keras
import time

import warnings
warnings.filterwarnings("ignore")

model = keras.models.load_model("/home/pi/k_model")
model.compile()

# run inference on the model
path2data = '/home/pi/testseqs.npz'

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
testData = testData.astype(np.float32)

#print("Reconstructed Output")
#print(np.shape(testData[0]))
#print(np.shape([input_np]))
#start = time.time()
model.predict(tf.convert_to_tensor([testData]))
#stop = time.time()
#print((stop-start)*1000)
#print(model(a))
#print("Input")
#print(testData[0][:6])
inference_time = []
res = np.zeros_like(testData)
print(np.shape(testData))
for i in range(100):
    in_tf = tf.convert_to_tensor([testData[i]])
    start = time.time()
    res[i] = model.predict(in_tf)
    stop = time.time()
    inference_time.append(stop-start)

    t = np.array(inference_time) * 1000
    def rmse(a,b):
      tmp = (a-b)**2
      tmp = np.sum(tmp, axis=1)
      return np.sqrt(tmp)/np.shape(a)[1]

print("Time per sample: {:.2f}ms".format(np.sum(t)/100))
print("Average RMSE: {:.7f}".format(rmse(testData, res).mean()))
