import flask
import tools
import file_engine
import bert_engine
import hashlib

app = flask.Flask(__name__)

with open('static/token', encoding='UTF-8') as f:
    token = f.read()


@app.route('/register', methods=['POST'])
def register():
    """Контроллер регистрации новых пользователей"""
    content = flask.request.get_json()

    if not tools.validate_reg_request(content):
        return "BAD request", 400
    if content['token'] != token:
        return "Invalid token", 403

    file_engine.register(content['user'], content['channel'])
    return "OK", 200


@app.route('/predict', methods=['POST'])
def predict():
    """Контроллер оценки полезности"""
    content = flask.request.get_json()

    if not tools.validate_pred_request(content):
        return "BAD request", 400
    if content['token'] != token:
        return "Invalid token", 403

    utility = []
    for text in content['text']:
        # hash = hashlib.md5(text.encode('utf-8')).hexdigest() # На случай подключения Redis
        p_text = bert_engine.extract(text)
        utility.append(file_engine.predict(content['user'], content['channel'], p_text))
    return flask.jsonify(**{'utility': utility}), 200


@app.route('/fit', methods=['POST'])
def fit():
    """Контроллер обучения пар пользователь:канал"""
    content = flask.request.get_json()

    if not tools.validate_fit_request(content):
        return "BAD request", 400

    p_text = []
    for text in content['text']:
        p_text.append(bert_engine.extract(text))
    file_engine.async_fit(content['user'], content['channel'], p_text, content['labels'])
    return "OK", 200


if __name__ == '__main__':
    app.run(debug=True)
