# evaluating-edge-accelerator
In this project two modern neural network accelerator are being evaluated by running inference using an auto encoder.

# Prerequisites

For the machine the NCS2 is connected, first we need to install Openvino. For an ubuntu machine use [these instructions for apt install](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html). 
Openvino is then installed to /opt/intel/openvino2021. You need to set the environment variables by calling 

```bash
source /opt/intel/openvino2021/bin/setupvars.sh
```

For the Google Coral Dev Board follow this getting started guide, if Mendel linux is not flashed on the board yet. Otherwise connect to it with 

```bash
mdt shell
```

# How to use

Folder pi contains benchmarking scripts for measuring with tensorflow (on a Pi or everywhere else), start with following command

```bash
python3 measure_pi.py
```

For measuring performance of the NCS2 or a CPU using openvino call
```bash
python3 measure_ncs2.py -m model.xml -d [MYRIAD/CPU]
```

Folder edge_tpu contains benchmarking scripts for measuring with tensorflow lite either on the tpu or on the ARM cores depending on the model your a using, start with following command

for calculation on the Edge TPU:

```bash
mdt shell
```
