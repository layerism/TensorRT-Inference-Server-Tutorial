

# TensorRT Inference Server 菜鸟教程

通过一个简单易懂，方便快捷的教程，部署一套完整的深度学习模型，一定程度可以满足部分工业界需求。对于不需要自己重写服务接口的团队来说，使用 tesorrt inference sever 作为服务，也足够了。

这里采取的案例是单目标检测的 centernet，SSD，YOLO 系列都比较古老了，虽然教程也比较多，但是都不够简洁而且相对思想比较老，稍微用点新的。

本教程使用的检测模型暂时不提供  model zoo，主要原因是官方 release 的 model 都带 DCN 模块，这个模块有 c++ 层面的库，作为初学者来说，部署起来非常不方便，大家可以根据 centernet 官方提供的代码，自行训练不带 DCN 的模型。

本教程使用的是 DLA34 网络作为例子，模型文件需要自行训练获取 pth，然后放置到 ./example/detection/network 下面



## 前言

对于绝大多数深度学习部署问题，总是包含如下的基本操作：前处理，神经网络计算，后处理

值得注意的是，每个前处理不仅需要完成数据解析，标准化等常见操作，还可能需要保存输入数据的一些整体信息，比如原始图像大小，字符串标注信息等，这些 meta 信息需要交给后处理用来做各种针对性的问题，对于 centernet 来说，这个 meta 信息就是仿射变换。

```python
nn_inputs, meta = preprocess(raw_image)
nn_outputs = model(nn_inputs)
result = postprocess(nn_outputs, meta)
```

本教程的实现路径如下：

1.  前处理采取 tensorflow 编写，包括图像解析，resize，计算仿射变换矩阵，标准化等，保存成 tensorflow pd 文件 
2.  神经网络部分是 torch，首先把 torch 的模型转换成 onnx，然后通过 onnx-simplifier 简化，接着交由 tensorRT 进行进一步优化，以及做 int8 量化。onnx-simplifier 的目的是为了更好地避免 onnx 到 tensorRT 的转换失败，但是，其并不能够百分百保证所有网络都能够被成功转换成 tensorRT，比如 torch 里面的 unsquezze 等 shape 层面的操作会有潜在问题，需要 model.py 里面改改。
3.  后处理是 torch 编写，然后转成 onnx，靠 onnx runtime 调度
4.  tensorRT 提供 ensemble 模式，可以联合调度 tensorflow 的 pd 文件，tensorRT plan 文件，onnx 格式文件，这样一来，可以把前处理，NN 计算，后处理都服务化，免除工程师搞复杂的编译工作和写 c++ 的工作，整个部署只需要写 python，特别通用高效，且没有竞争力



## 服务端搭建

```sh
docker pull nvcr.io/nvidia/tensorrtserver:19.12-py3
```

注意，这里面需要 nvidia 驱动版本大于 418 才行，Tesla 系列显卡可以稍微降低要求，cuda 版本要求是 10.1，详细配置参考：

https://docs.nvidia.com/deeplearning/sdk/inference-release-notes/rel_19-12.html#rel_19-12



## 客户端搭建

```sh
docker pull nvcr.io/nvidia/tensorrtserver:19.12-py3-clientsdk
```

理论上来说，grpc 接口不依赖系统环境，没必要靠 docker 启动客户端，docker run 上述镜像以后，把 /workspace/install/python/tensorrtserver-1.9.0-py2.py3-none-linux_x86_64.whl 的安装文件取出来，直接在任意一台机器 pip install 便可

```sh
# docker run --rm nvcr.io/nvidia/tensorrtserver:19.12-py3-clientsdk /bin/bash 
# copy `/workspace/install/python/tensorrtserver-1.9.0-py2.py3-none-linux_x86_64.whl` file to any linux machine
# run the following commad
pip install tensorrtserver-1.9.0-py2.py3-none-linux_x86_64.whl
```

对于 c++ 来说，把 client 端的 SDK 抠下来找个地方编译自己的文件即可，这里比较烦，暂时不做例子。



## Inference Server Backend 安装

安装各种 backend，用于生成如下转换格式： 

-   onnx
-   tensorRT
-   tensorflow
-   pytorch

安装 TensorRT-6.0.1.5，请参考 https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html

安装其它 backend 库，目前只需要 python 端的即可：

```sh
pip install onnx==1.6.0 onnxruntime==1.1.0 onnx-simplifier==0.2.2
pip install tensorflow-gpu==1.5.0
pip install torch==1.3.0 torchvision==0.4.1
pip install opencv-python pillow pycuda
```

安装转换工具：

```sh
cd backend
python setup.py install
```



## 开始教程

安装教程内的转换脚本和客户端接口：

```sh
cd backend
python setup.py install
cd client_py
python setup.py install
```

执行教程的 example，这个 example 会生成完整的 model-repo，剩下交给 tensorRT inference server 

```sh
cd example/detection
./convert.sh
```

启动服务：

```sh
./start.sh
```

使用 client：

```sh
cd example/detection
./client.sh
```



## python 客户端使用

单步调度举例：

```python
from trt_client import client
import numpy as np

raw_image = open("./xxx.jpg", "rb").read()
raw_image = np.array([raw_image], dtype=bytes)

runner = client.Inference(
	url="xx.xxx.xxx.xxx:7001", # grpc
	model_name="detection",
	model_version="1"
)
results = runner.run(input={"raw_image": raw_image})
print(results)
```

异步非阻塞调度举例：

```python
from trt_client import client
import numpy as np

runner = client.Inference(
	url="xx.xxx.xxx.xxx:7001", # grpc
	model_name="detection",
	model_version="1"
)

for i in range(10):
    raw_image = open("./{}.jpg".format(i), "rb").read()
	raw_image = np.array([raw_image], dtype=bytes)
	results = runner.async_run(
        input={"raw_image": raw_image}, 
        input_id="image_{}".format(i)
   	)
for i in range(10):
    input_id, results = runner.get(block=True)
```

