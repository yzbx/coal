import time
from flask import Flask, Response, after_this_request

app = Flask(__name__)

@app.route('/')
def main():
    return '''<div>start</div>
    <script>
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/test', true);
        xhr.onreadystatechange = function(e) {
            var div = document.createElement('div');
            div.innerHTML = '' + this.readyState + ':' + this.responseText;
            document.body.appendChild(div);
        };
        xhr.send();
    </script>
    '''

@app.route('/test')
def test():
    def generate():
        app.logger.info('request started')
        for i in range(10):
            time.sleep(1)
            yield str(i)
        app.logger.info('request finished')
        yield ''
        
        @after_this_request
        def after(response):
            time.sleep(1)
            print('hello world')
        
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run('0.0.0.0', 8080, True)