# 部署流程
代码push到github，镜像push到阿里云，服务器拉镜像运行
# 基本概念
## 路由
路由是 URL 到 Python 函数的映射。Flask 允许你定义路由，使得当用户访问特定 URL 时，Flask 会调用对应的视图函数来处理请求。
```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Home Page!'

@app.route('/about')
def about():
    return 'This is the About Page.'
```
## 视图函数
视图函数是处理请求并返回响应的 Python 函数。它们通常接收请求对象作为参数，并返回响应对象，或者直接返回字符串、HTML 等内容。
```py
from flask import request

@app.route('/greet/<name>')
def greet(name):
    return f'Hello, {name}!'
```
## 请求对象
请求对象包含了客户端发送的请求信息，包括请求方法、URL、请求头、表单数据等。Flask 提供了 request 对象来访问这些信息。
```py
from flask import request

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    return f'Hello, {username}!'
```
## 响应对象
响应对象包含了发送给客户端的响应信息，包括状态码、响应头和响应体。Flask 默认会将字符串、HTML 直接作为响应体。
```py
from flask import make_response

@app.route('/custom_response')
def custom_response():
    response = make_response('This is a custom response!')
    response.headers['X-Custom-Header'] = 'Value'
    return response
```
make_response：创建一个自定义响应对象，并设置响应头 X-Custom-Header。

## 跨域问题？
跨域问题只是浏览器的限制，和后端无关，只要后端设置允许跨域，前端就能正常访问接口！

# 后端需要
## 基层方面，能否正确解析请求体
1. Content-Type校验，检查请求头是否为application/json
2. JSON解析，解析出错返回400
## 数据结构与类型：Schema校验
1. 必填字段
2. 字段类型
3. 范围
4. 默认值
## 业务层满：更细粒度的校验
1. 字符串长度
2. 格式校验
3. 数值范围
4. 依赖关系
5. 去重/长度限制
## 安全层面（需要补充）
## 授权于认证（需要补充）
## 返回有好的错误信息
1. 统一错误格式