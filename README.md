# ONNX TO TRT
Pre-requisites:
 - Install Custom Detectron2 for SIA model
```
cd /home/eslab/workspace/onnx_to_trt/libs
python -m pip install -e detectron2
```
Conversion ONNX to TRT
- Fisrt, move the onnx file in to '/home/eslab/workspace/onnx_to_trt/onnx_files'
```
FP32 : python3 -m detection.onnx_to_trt --model-path=$onnx_file_path --save-path=$engine_path
FP16 : python3 -m detection.onnx_to_trt --model-path=$onnx_file_path --save-path=$engine_path --data-type fp16
INT8 : python3 -m detection.onnx_to_trt --model-path=$onnx_file_path --save-path=$engine_path --data-type int8
MIX : python3 -m detection.onnx_to_trt --model-path=$onnx_file_path --save-path=$engine_path --data-type mix
```

Test TRT engine file
```
./trtexec --loadEngine=$engine_path
```

# Deepstream with TRT
Move the engine file to '/home/eslab/workspace/deepstream_SIA/engine'
Open 'deepstream_app_config_fasterRCNN.txt' config file and change the engine path

```
[primary-gie]
enable=1
gpu-id=0
batch-size=1
gie-unique-id=1
interval=0
labelfile-path=labels.txt
model-engine-file=engine/fastest.engine
config-file=config_infer_primary_fasterRCNN.txt
nvbuf-memory-type=0

```
