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