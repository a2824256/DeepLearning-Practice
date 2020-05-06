# 网络层介绍
### Inception block(出自inception network)<br>
#### 参考:
https://baijiahao.baidu.com/s?id=1601882944953788623&wfr=spider&for=pc

#### 作用:
对传统卷积层进行分解<br/>
#### 三种分解模块
##### 模块A<br/>
![model-a](./material_image/inception-network/model-a.jpg)<br/>
##### 模块B, 空间非对称卷积<br/>
![model-b](./material_image/inception-network/model-b.jpg)<br/>
##### 模块C<br/>
![model-c](./material_image/inception-network/model-c.jpg)<br/>
##### inception-graph(查看分解模块在网络中的位置)
![inception-graph](./material_image/inception-network/inception-graph.jpg)<br/>
Inception的stem是指初始网络层
- figure 5对应model a
- figure 6对应model b
- figure 7对应model c

###  