
## Model conversion and quantization
The entire model conversion process can be summarized as follows: pytorch-->onnx-->tmfile-->uint tmfile
1) Converting pytorch model to onnx model (.onnx):
```
# Ataach the docker sheel to the yolox docker container and Go to yolox directory and run the following command
python tools/export_onnx.py --output-name yolox_nano_spot.onnx -f exps/default/yolox_nano_rwl.py -c weights/yolox_nano_rwl_e300_spot_dataset.pth -o 12
```
where

	-f is the experiment file for the yolox model containing details about model training hyperparameters, high-level model architecture details, etc.

	-c is the path to pytorch weights

	-o is the opset that you want for onnx model conversion (I used opset 12 for successfull conversion)

2) Converting onnx model (onnx) to tmfile:
```
./../../Tengine-Convert-Tools/build/install/bin/convert_tool -f onnx -m yolox_nano_spot.onnx -o yolox_nano_spot.tmfile
```
argument options:

	-h    help            show this help message and exit

        -f    input type      framework (onnx, tensorflow, etc)

        -p    input structure path to the network structure of input model(*.prototxt, *.symbol, *.cfg)

        -m    input params    path to the network params of input model(*.caffemodel, *.params, *.weight, *.pb, *.onnx, *.tflite)

        -o    output model    path to output fp32 tmfile 

3) Converting tmfile to uint tmfile
```
./../../tengine-lite/quant_tool_precompiled/quant_tool_uint8 -m yolox_nano_spot.tmfile -i datasets/COCO/test2017/ -o yolox_nano_spot_uint8.tmfile  -g 3,416,416 -w 128,128,128
```
argument options:

        -h    help            show this help message and exit

	-m    input model     path to input float32 tmfile

	-i    image dir       path to calibration images folder

	-f    scale file      path to calibration scale file

	-o    output model    path to output uint8 tmfile

	-a    algorithm       the type of quant algorithm(0:min-max, 1:kl, default is 0)

	-g    size            the size of input image(using the resize the original image,default is 3,224,224)

	-w    mean            value of mean (mean value, default is 104.0,117.0,123.0)

	-s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)

	-b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)

	-c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)

	-y    letter box      flag which indicates that letter box process image is necessary(maybe using for YOLOv3/v4, 0:OFF, 1:ON, default is 0)

	-k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)

	-t    num thread      count of processing threads(default is 1)        

## Forward inference of tmfile
```
./build/install/bin/tm_yolox -m ../neural_networks_docker/YOLOX/weights/yolox_nano_spot.tmfile -i ../yolov3_quantization_aware/dataset/test/images/P10009.png -r 1 -t 1
```