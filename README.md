## TensorRT custom plugin -> Prpool implement
## TensorRT 8.5.3.1
TensorRT8.5.3.1 install refer to ![How to install complate TensorRT and set environment PATH?](https://uin1gam3t89.feishu.cn/docx/M3vydV5EZop8u8x8wj7csfSKnOg)
### First use gcc make 
Run make clean and make in the ./custom_prpool_plugin and generate .so file , just as follow:
![6a06516e09033288bea552873cdd7f4](https://github.com/USCT-YQJ/custom_prpool_plugin/assets/96329803/3842d10c-d4cc-4955-8fac-6c40f636751a)
### Second use TensorRT python API generate .engine file
Run CUDA_VISIBLE_DEVICES=7 python src/python/unit_test_customPrpool.py, just as follow:
![4ee1b0ebf741a02c27e722ae8006125](https://github.com/USCT-YQJ/custom_prpool_plugin/assets/96329803/25999646-4db1-4e39-9fd2-6e295adbf4e0)
### Finally use TensorRT python API run .engine file
Run CUDA_VISIBLE_DEVICES=7 python src/python/torch2onnx2trt_tgthead.py, just as follow:
![dc65ba480c9fc81d2b3ded85399c72f](https://github.com/USCT-YQJ/custom_prpool_plugin/assets/96329803/0988d876-1fe8-4e34-9ddd-cac08bab0071)
## Note:
this implement is checked to insure correct, as follow:
![image](https://github.com/USCT-YQJ/custom_prpool_plugin/assets/96329803/cd0c1f93-5bdc-4905-ba34-0de048e1750c)
