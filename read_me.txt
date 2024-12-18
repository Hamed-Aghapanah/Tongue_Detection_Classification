با سلام

import torch
print(torch.cuda.is_available())
python train.py --device cpu
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
pip install --upgrade torch torchvision
import os
os.environ['OMP_NUM_THREADS'] = '1'
pip install --upgrade torch torchvision
nvidia-smi
$env:KMP_DUPLICATE_LIB_OK="TRUE"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
pip install --upgrade numpy torch torchvision
pip install keras-preprocessing libclang tensorflow-io-gcs-filesystem protobuf==3.19.0
pip install --upgrade tensorflow keras-preprocessing libclang tensorflow-io-gcs-filesystem protobuf
pip install pipdeptree
pip uninstall onnx
pip install --upgrade tensorflow protobuf onnx
pip install tensorflow onnx

python train.py --cfg cfg/training/yolov7-tiny.yaml --weights best_tongue.pt --data data/Tongue.yaml --batch-size 8 --epochs 600 --workers 0


python detect.py --weights best_tongue.pt --source 0 

