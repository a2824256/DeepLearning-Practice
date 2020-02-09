## 学习资料 - 卷积等算法
https://github.com/a2824256/DeepLearning-Practice/blob/master/material.md
## 神经网络
#### LeNet-5
Pytorch版本: <br/>
https://github.com/a2824256/DeepLearning-Practice/blob/master/notebooks/LeNet.ipynb
<br/>
Paper下载地址:<br/>
https://www.researchgate.net/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition 
<br/>
#### FCN(实现中)
Pytorch版本: <br/>
https://github.com/a2824256/DeepLearning-Practice/blob/master/notebooks/FCN.ipynb

#### VGG16(实现中)
Pytorch版本: <br/>
https://github.com/a2824256/DeepLearning-Practice/blob/master/notebooks/VGG16.ipynb
## TensorFlow入门
https://github.com/a2824256/DeepLearning-Practice/tree/master/notebooks/tf_teach
## 环境配置
> 推荐使用conda
### conda添加国内源
> 清华源
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
```
> pytorch源
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```
> 中科大源
```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
```
#### 设置搜索时显示通道地址
```conda config --set show_channel_urls yes```
#### 部分库conda安装指令
- 安装cv2库： <br/>
```conda install --channel https://conda.anaconda.org/menpo opencv```
- 安装pytorch以及torchvision库： <br/>
```conda install pytorch torchvision -c pytorch```
#### pip使用清华源安装TensorFlow-GPU
pip install -i https://pypi.douban.com/simple/ tensorflow-gpu==1.15

## 相关资料
### 关于baseline和benchmark的说明
https://www.zhihu.com/question/22529709
### end to end(端对端)模型的理解
- 输入端（数据）-> 输出端（结果）,问题使用单个模型就可解决，例如使用单个CNN能实现图像识别，不像自然语言处理需要使用多个模型解决问题
### tensorflow中SAME和VALID之间的区别
> 这两个概念原来是出自计算机视觉里的，目的是对图片的扩展。现在到了图像卷积这里也是如此。我们在卷积的时候卷积核的移动往往会跳出图片或者丢弃一小部分像素点。从结果上影响上来说，二者好像没多大影响，你可以这么认为，一张图片上的边缘像素点一般都不会有重要特征的，除非是抓拍拍下的对象偏移，处在了边缘。既然这样那么我们还是有必要弄清楚二者的计算关系。

> SAME：这种padding方法法在tensorflow中就是为了保证输入和输出的结构一致。那么几个padding能做到就看图片尺度了。这种情况一般步长都为1：

官方API给出的计算方法如下：

```
out_height = ceil(float(input_height)/float(strides[0]))
out_width = ceil(float(input_weight)/float(strides[1]))
```

> VALID:这种padding就是没有padding，就是在卷积核不长不足或者超出的部分直接舍去，这样得到的输出相比输入尺寸较小。当然VALID也可以实现输出相同，那就是如AlexNet中采用一个group，将其切开交替卷积，这样做得到的结果和加入padding一样，即能够保持输入输出一种。不过参与卷积的像素有所不同，即不全为零。这也是个了不起的trick。

官方API给出的计算方法如下：

```
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
```
