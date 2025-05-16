```
docker run --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model densenet_onnx
```


```
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:25.02-py3-sdk /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```
