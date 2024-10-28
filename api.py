from flask import Flask, request
from flask_cors import CORS

from analyzerService import AnalyzerService

app = Flask(__name__)
CORS(app)

service = AnalyzerService()


@app.route('/api/v1/analyze', methods=['GET'])
def analyze():
    text_to_analyze = request.args.get('txt')
    results = service.analyze_emotion(text_to_analyze)
    return results


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
