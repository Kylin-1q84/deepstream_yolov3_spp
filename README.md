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
