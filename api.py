import os
import sys
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from database import DatabaseManager
# from chat import ChatBot
# from agent import Agent
from agent2 import MultiAgentSystem as Agent

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

db = DatabaseManager()

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Helper function for OPTIONS responses
    def options_response():
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        return response

    # Add OPTIONS handler for all API routes
    @app.route('/api/<path:path>', methods=['OPTIONS'])
    def handle_options(path):
        return options_response()

    # Root Route with API Documentation
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        return jsonify({
            "message": "Chatbot API",
            "available_endpoints": {
                "Authentication": [
                    {"route": "/api/register", "method": "POST", "description": "Register a new user"},
                    {"route": "/api/login", "method": "POST", "description": "User login"}
                ],
                "Chat Sessions": [
                    {"route": "/api/sessions", "method": "GET", "description": "Retrieve user's chat sessions"},
                    {"route": "/api/sessions", "method": "POST", "description": "Create a new chat session"},
                    {"route": "/api/sessions/<session_id>", "method": "DELETE", "description": "Delete a specific chat session"}
                ],
                "Chat Messages": [
                    {"route": "/api/sessions/<session_id>/messages", "method": "GET", "description": "Retrieve messages for a session"},
                    {"route": "/api/sessions/<session_id>/messages", "method": "POST", "description": "Save a new message to a session"},
                    {"route": "/api/sessions/<session_id>/generate", "method": "POST", "description": "Generate and save the bot response to a session"}
                ]
            }
        }), 200

    # Register Route
    @app.route('/api/register', methods=['POST', 'OPTIONS'])
    def register():
        if request.method == 'OPTIONS':
            return options_response()
        
        data = request.get_json()

        # Validate input
        if not data or not all(k in data for k in ('username', 'password', 'email')):
            return jsonify({"error": "Missing required fields"}), 400
               
        # db = DatabaseManager()
        
        # Attempt registration
        if db.register_user(data['username'], data['password'], data['email']):
            return jsonify({"message": "User registered successfully"}), 201
        else:
            return jsonify({"error": "Username or email already exists"}), 409

    # Login Route
    @app.route('/api/login', methods=['POST', 'OPTIONS'])
    def login():
        if request.method == 'OPTIONS':
            return options_response()
        
        data = request.get_json()
        
        # Validate input
        if not data or not all(k in data for k in ('username', 'password')):
            return jsonify({"error": "Missing username or password"}), 400
        
        
        # Verify user
        # db = DatabaseManager()
        if db.verify_user(data['username'], data['password']):
            return jsonify({
                "message": "Login successful", 
                "username": data['username']
            }), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    # Sessions Route - GET (Retrieve all sessions for a user)
    @app.route('/api/sessions', methods=['GET', 'OPTIONS'])
    def handle_sessions_get():
        if request.method == 'OPTIONS':
            return options_response()
        
        username = request.args.get('username')
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        # db = DatabaseManager()
        sessions = db.get_user_chat_sessions(username)
        
        # session_list = [
        #     {"session_id": session_id, "session_name": session_name, "created_at": created_at}
        #     for session_id, session_name, created_at in sessions
        # ]
        
        return jsonify(sessions), 200

    # Sessions Route - POST (Create a new session)
    @app.route('/api/sessions', methods=['POST', 'OPTIONS'])
    def handle_sessions_post():
        if request.method == 'OPTIONS':
            return options_response()

        data = request.get_json() or {}

        username = data.get('username')
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        session_name = data.get('session_name', None)
        
        # db = DatabaseManager()
        session_id = db.create_chat_session(username, session_name)
        
        return jsonify({"message": "Session created successfully", "session_id": session_id}), 201

    # Delete a Session Route
    @app.route('/api/sessions/<session_id>', methods=['DELETE', 'OPTIONS'])
    def delete_session(session_id):
        if request.method == 'OPTIONS':
            return options_response()
        
        # db = DatabaseManager()
        success = db.delete_chat_session(session_id)
        
        if success:
            return jsonify({"message": "Session deleted successfully"}), 200
        else:
            return jsonify({"error": "Session not found"}), 404

    # Get Messages for a session
    @app.route('/api/sessions/<session_id>/messages', methods=['GET', 'OPTIONS'])
    def get_messages(session_id):
        if request.method == 'OPTIONS':
            return options_response()
        
        # db = DatabaseManager()
        messages = db.get_chat_messages(session_id)
        
        return jsonify(messages), 200

    # Save a Message to a session
    @app.route('/api/sessions/<session_id>/messages', methods=['POST', 'OPTIONS'])
    def save_message(session_id):
        if request.method == 'OPTIONS':
            return options_response()
        
        data = request.get_json()

        if not data or 'username' not in data or 'message' not in data:
            return jsonify({"error": "Missing username or message field"}), 400
        
        # db = DatabaseManager()
        db.save_chat_message(session_id, data['username'], data['message'])
        
        return jsonify({"message": "Message saved successfully"}), 201

    # Generate and Save Bot Response for a session
    # @app.route('/api/sessions/<session_id>/generate', methods=['POST', 'OPTIONS'])
    # def generate_bot_response(session_id):
    #     if request.method == 'OPTIONS':
    #         return options_response()
        
    #     data = request.get_json()

    #     if not data or 'message' not in data:
    #         return jsonify({"error": "Missing message field"}), 400

    #     user_message = data['message']
        
    #     # db = DatabaseManager()
    #     # chatbot = ChatBot(data.get('username'))  # Assuming session_id is the username here
    #     chatbot = Agent(data.get('username'))
        
    #     # Generate the response
    #     bot_response = ""
    #     try:
    #         for chunk in chatbot.generate_bot_response(session_id,user_message):
    #             bot_response += chunk
    #     except Exception as e:
    #         return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500
        
    #     # Save the response message to the session
    #     db.save_chat_message(session_id, "bot", bot_response)
        
    #     return jsonify({"bot_response": bot_response}), 201
    
    @app.route('/api/sessions/<session_id>/generate', methods=['POST', 'OPTIONS'])
    def generate_bot_response(session_id):
        if request.method == 'OPTIONS':
            return options_response()
        
        data = request.get_json()

        if not data or 'message' not in data or 'username' not in data:
            return jsonify({"error": "Missing required fields"}), 400

        user_message = data['message']
        username = data['username']
        
        try:
            # Initialize agent with username
            agent = Agent(username)
            
            # Generate response
            bot_response = agent.generate_bot_response(session_id, user_message)
            
            if bot_response:
                # Save the response message to the session
                db.save_chat_message(session_id, "bot", bot_response)
                return jsonify({"bot_response": bot_response}), 201
            else:
                return jsonify({"error": "Failed to generate response"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

    # Error Handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({"error": "Internal server error"}), 500

    return app

# For Vercel or local
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)