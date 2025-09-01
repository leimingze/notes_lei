# docker介绍
docker镜像：你项目的”快递包裹“（含所用得到的东西）
docker容器：你项目的”快递员“（含你项目的”快递包裹“）
服务器主要有docker，就像快递柜，包裹随便投，随时用！
# 阿里云命名空间和仓库有什么区别
一个命名空间下可以有很多的仓库。例如
```py
假设你在阿里云注册时叫 alice，你有两个项目：
Flask项目：myflaskapp
Scrapy项目：myscrapyspider
你可以这么设计：
命名空间：alice
    仓库1：myflaskapp
        镜像tag: latest、v1.0、v2.0
    仓库2：myscrapyspider
        镜像tag: latest、v1.0
```
# 详细步骤
## 配置阿里云加速器
docker默认要从国外docker hub拉镜像，速度很慢，所以需要配置阿里云加速器。进入阿里云加速器获取页面
```Bash
https://cr.console.aliyun.com/cn-beijing/instance/mirrors
```

## 本地构建Docker镜像
```Bash
docker build -t crpi-2zb1inyra2trt0wr.cn-beijing.personal.cr.aliyuncs.com/你的命名空间/你的仓库名:tag .
```
### dockerfile
## 登录阿里云仓库
```Bash
docker login --username=leimingze crpi-2zb1inyra2trt0wr.cn-beijing.personal.cr.aliyuncs.com
```
## 推送镜像到阿里云仓库
```Bash
docker build -t crpi-2zb1inyra2trt0wr.cn-beijing.personal.cr.aliyuncs.com/你的命名空间/你的仓库名:tag .
```
## 从阿里云仓库拉取镜像
```Bash
 docker pull crpi-2zb1inyra2trt0wr.cn-beijing.personal.cr.aliyuncs.com/lei_ali/flask:[tag]
```
## 运行镜像并使用目录挂载，让容器使用服务器上的代码
```Bash
docker run -d \
  -p 5000:5000 \
  -v /home/admin/flaskproject:/app \
  --name my-flask-app \
  your-image-name
```
✅ -d 表示后台运行
✅ -p 5000:5000 将宿主机 5000 端口映射到容器 5000 端口
✅ -v /home/admin/flaskproject:/app 表示把服务器 /home/admin/flaskproject 挂载到容器的 /app，容器内会实时读取这个目录下的代码
✅ --name my-flask-app 给容器起个名字方便管理
✅ your-image-name 替换成你实际镜像名或 ID
## 继续运行当前镜像
```Bash
docker start my-flask-app
```
## 查看镜像列表
```Bash
docker images
```
## 查看当前正在运行的镜像
```Bash
docker ps
```
## 查看所有镜像
```Bash
docker ps -a
```
## 删除镜像
```Bash
docker rmi 镜像id
```
## 停止运行的容器
```Bash
docker stop 容器id
```