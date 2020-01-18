

# TensorRT Inference Server 初学者教程

通过一个简单易懂，方便快捷的教程，部署一套完整的深度学习模型，一定程度可以满足部分工业界需求。

对于不需要自己重写服务接口的团队来说，使用 tesorrt inference sever 作为服务，也足够了。

这里采取的例子是单目标检测的 centernet 的部署，SSD，YOLO 系列都比较古老了，虽然教程也比较多，但是都不够简洁。

本教程使用的检测模型暂时不提供  model zoo



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
conda install openssl=1.1.1
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



## 安装教程自带的工具

```sh
cd backend
python setup.py install
cd client_py
python setup.py install
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
```

异步非阻塞调度举例：

```python
from trt_client import client
import numpy as np

runner = client.Inference(
	url="xx.xxx.xxx.xxx:7001", # grpc
	model_name="face-det",
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

