#!/bin/bash

# ldd /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
/usr/src/tensorrt/bin/trtexec --onnx=./model/backbone.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/backbone.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed 
