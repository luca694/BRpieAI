from flask import Flask, jsonify
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

print("🚀 STARTING FLASK SERVER...")
print("📍 This should appear immediately when you run the script")

@app.route('/test', methods=['GET'])
def test():
    print("✅ /test endpoint was called!")
    return jsonify({
        "status": "working", 
        "message": "Flask server is running!",
        "endpoint": "/test"
    })

@app.route('/process-face', methods=['POST'])
def process_face():
    print("✅ /process-face endpoint was called!")
    return jsonify({
        "success": True,
        "recognized": True,
        "confidence": 92.5,
        "user_id": "kb",
        "user_name": "Khoo Ben",
        "user_st": "2291",
        "grade": "11",
        "class": "5/1",
        "message": "Test recognition successful"
    })

@app.route('/')
def home():
    return """
    <html>
        <body style="font-family: Arial; margin: 40px;">
            <h1>✅ Flask Server is Running!</h1>
            <p>Test endpoints:</p>
            <ul>
                <li><a href="/test">GET /test</a> - Health check</li>
                <li>POST /process-face - Face recognition</li>
            </ul>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("🎯 Starting Flask development server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("⏹️  Press CTRL+C to stop the server")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)