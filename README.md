# ONNX TO TRT


Pre-requisites:
 - Install Custom Detectron2 for SIA model
```
cd /home/eslab/workspace/onnx_to_trt/libs
python -m pip install -e detectron2

```






Compile the custom library:

```
  # Based on the API to use 'NvDsInferCreateModelParser' or 'NvDsInferCudaEngineGet'
  # set the macro USE_CUDA_ENGINE_GET_API to 0 or 1 in
  # nvdsinfer_custom_impl_Yolo/nvdsinfer_yolo_engine.cpp

  # Export correct CUDA version (e.g. 10.2, 10.1)
  $ export CUDA_VER=10.2
  $ make -C nvdsinfer_custom_impl_Yolo
```
--------------------------------------------------------------------------------
Run the sample:
The "nvinfer" config file config_infer_primary_yolo.txt specifies the path to
the custom library and the custom output parsing function through the properties
"custom-lib-path" and "parse-bbox-func-name" respectively.
The first-time a "model_b1_int8.engine" would be generated as the engine-file

- With deepstream-app
  $ deepstream-app -c deepstream_app_config_yoloV3.txt
  $ deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
  $ deepstream-app -c deepstream_app_config_yoloV2.txt
  $ deepstream-app -c deepstream_app_config_yoloV2_tiny.txt
  $ LD_PRELOAD=<path-to-TRT-OSS-libnvinfer_plugin.so.7.0.0> deepstream-app -c deepstream_app_config_yolo_tlt.txt
