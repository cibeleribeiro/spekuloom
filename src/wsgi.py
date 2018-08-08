from bottle import route, redirect, run, view
import os
import bottle

base_path = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_path, 'view')
bottle.TEMPLATE_PATH.insert(0, template_dir)


@route('/')
@view('index')
def index():
    # redirect('view/index.html')
    return {}


if __name__ == '__main__':
    run(host='localhost', port=8080)
