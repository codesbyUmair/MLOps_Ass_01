from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    """
    Home route that returns a welcome message.
    """
    return jsonify({
        "message": "Welcome to the DevOps Pipeline Demo Application!",
        "status": "healthy"
    }), 200

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return jsonify({
        "status": "ok",
        "message": "Application is running smoothly"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)