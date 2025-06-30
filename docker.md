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
## 本地：登录阿里云镜像仓库
