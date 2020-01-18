

# TensorRT Inference Server 部署教程



## Pull 服务的镜像

```sh
docker pull nvcr.io/nvidia/tensorrtserver:19.12-py3
docker pull nvcr.io/nvidia/tensorrtserver:19.12-py3-clientsdk
```

注意，这里面需要 nvidia 驱动版本大于 418 才行，Tesla 系列显卡可以稍微降低要求，cuda 版本要求是 10.1，详细配置参考：

https://docs.nvidia.com/deeplearning/sdk/inference-release-notes/rel_19-12.html#rel_19-12



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



## python 客户端使用

安装必要依赖：

```sh
curl http://ai-zzzc.s3.360.cn/package/tensorrtserver-1.9.0-py2.py3-none-linux_x86_64.whl --output ./tmp/tensorrtserver-1.9.0.whl
pip install ./tensorrtserver-1.9.0.whl
conda install openssl=1.1.1
rm -rf ./tmp/tensorrtserver-1.9.0.whl
```

安装 trt_client 客户端：

```sh
cd client_py
python setup.py install
```

单步调度举例：

```python
from trt_client import client
import numpy as np

raw_image = open("./xxx.jpg", "rb").read()
raw_image = np.array([raw_image], dtype=bytes)

runner = client.Inference(
	url="10.160.168.155:7001",
	model_name="face-det",
	model_version="1"
)
results = runner.run(input={"raw_image": raw_image})
```

异步非阻塞调度举例：

```python
from trt_client import client
import numpy as np
import multiprocessing as mp

runner = client.Inference(
	url="10.160.168.155:7001",
	model_name="face-det",
	model_version="1"
)

queue = mp.Queue()
for i in range(10):
    raw_image = open("./{}.jpg".format(i), "rb").read()
	raw_image = np.array([raw_image], dtype=bytes)
	results = runner.async_run(
        input={"raw_image": raw_image}, 
        input_id="image_{}".format(i), 
        result_queue=queue
   	)
for i in range(10):
    input_id, results = runner.get(queue, block=True)
```

