from flask import Flask, request, jsonify

app = Flask(__name__)

#@app.route('/webhook', methods=['POST'])
#def webhook():
#    data = request.json  # Get the JSON data from the POST request
#    event_type = request.headers.get('X-GitHub-Event')  # Get the event type
#
#    # Perform actions based on the event_type and data
#    # For example, you can trigger builds, send notifications, etc.
#
#    return '', 200  # Respond with a success status

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers['Content-Type'] == 'application/json':
        data = request.json
        # Process the JSON data here
        return jsonify(success=True)
    else:
        return jsonify(success=False, error='Invalid content type'), 415

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
