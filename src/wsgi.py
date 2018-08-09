from bottle import route, run, template, view, redirect, static_file, get
import os
import bottle
base_path = os.path.abspath(os.path.dirname(__file__))
bananas_path = os.path.join(base_path, 'view')
bottle.TEMPLATE_PATH.insert(0, bananas_path)
img_dir = os.path.join(base_path, '_static')
css_dir = os.path.join(base_path,'view/css' )
@route('/')


@view('index')
def index():
    #return template('<b>Hello {{name}}</b>!', name=name)
    #redirect('view/index.html')
    return {}

@get("/_static/<filepath:re:.*\.(png|jpg|svg|gif|ico)>")
def img(filepath):
        return static_file(filepath, root=img_dir)

@get("/css/<filepath:re:.*\.css>")
def ajs(filepath):
        return static_file(filepath, root=css_dir)

run(host='localhost', port=8080)
