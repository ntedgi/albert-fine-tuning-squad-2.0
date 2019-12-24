import os
import json

from flask import Flask, jsonify, request

from flask_cors import CORS
from logzero import logger
from model import AlbertModelForQuestionAnswering


# input data example
# {"question":"Who was Jim Henson?", "paragraphs":[ {"id": 1, "text": "Jim Henson has a nice car. Jim Henson was a nic puppet"} , {"id":2,"text": "All the 2023(GoF) design patterns implemented inJavascript."}]}

def create_app(config=None):
    path_to_model = './model'
    model = AlbertModelForQuestionAnswering(path_to_model)
    app = Flask(__name__)

    app.config.update(dict(DEBUG=False))
    app.config.update(config or {})

    CORS(app)

    @app.route("/predict", methods=['GET', 'POST'])
    def predict_controller():
        result = []
        try:
            payload = json.loads(request.data)
            paragraphs = payload['paragraphs']
            question = payload['question']
            if paragraphs is None or len(paragraphs) == 0 or question is None or len(question) < 10:
                return jsonify({"error": "invalid arguments", "status": 500})
        except:
            logger.info("invalid arguments")
        try:
            result = model.predict(question, paragraphs)
        except:
            return jsonify({"error": "exception thrown in model", "status": 500})
        return jsonify({"result": result})

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app = create_app()
    app.run(host="0.0.0.0", port=port)
