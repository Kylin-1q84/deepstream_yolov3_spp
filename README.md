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
model-engine-file=engine/fastest.engine # hear
config-file=config_infer_primary_fasterRCNN.txt
nvbuf-memory-type=0
```

The defalut input image size is 1024,  ( Video has the 1536x1536, It is resize to 1024 )   
If you want change, modify config file 'deepstream_app_config_fasterRCNN.txt'
```
[tiled-display]
enable=0
rows=1
columns=1
width=1024 #hear
height=1024 #hear
gpu-id=0
nvbuf-memory-type=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=-1
## Set muxer output width and height
width=1024 #hear
height=1024 #hear
nvbuf-memory-type=0
```

In the defalut setting, Yon can only use the file input,   
If you want to change cam, modify this
```
[source0]
enable=1
num-sources=2
type=1
camera-width=1920
camera-height=1080
camera-fps-n=65
camera-fps-d=1
camera-csi-sensor-id=1
camera-v4l2-dev-node=0
drop-frame-interval=5
```

If you need more configuration, you can find this link   
https://docs.nvidia.com/metropolis/deepstream/5.0.1/dev-guide/index.html
