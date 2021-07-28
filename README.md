# evaluating-edge-accelerator
In this project two modern neural network accelerator are being evaluated by running inference using an auto encoder.

# How to use

Folder pi contains benchmarking scripts for measuring with tensorflow (on a Pi or everywhere else), start with following command

```python
python3 measure_pi.py
```

For measuring performance of the NCS2 or a CPU using openvino call
```python
python3 measure_ncs2.py -m model.xml -d [MYRIAD/CPU]
```
