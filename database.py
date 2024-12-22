from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
import hashlib
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv('MONGO_URI')

class DatabaseManager:
    def __init__(self, mongo_uri=mongo_uri, db_name="chatbot"):
        # Initialize MongoDB connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        # MongoDB collections
        self.users_col = self.db["users"]
        self.sessions_col = self.db["sessions"]
        self.messages_col = self.db["messages"]

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, email):
        # Check if username or email already exists in MongoDB
        if self.users_col.find_one({"$or": [{"username": username}, {"email": email}]}):
            return False
        
        # Create new user
        hashed_password = self.hash_password(password)
        user_data = {
            'username': username,
            'password': hashed_password,
            'email': email
        }
        
        try:
            self.users_col.insert_one(user_data)
            return True
        except DuplicateKeyError:
            return False

    def verify_user(self, username, password):
        user = self.users_col.find_one({"username": username})
        if user and user['password'] == self.hash_password(password):
            return True
        return False

    def create_chat_session(self, username, session_name=None):
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Default session name if none provided
        if not session_name:
            session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session_data = {
            'username': username,
            'session_id': session_id,
            'session_name': session_name,
            'created_at': datetime.now().isoformat()
        }
        
        self.sessions_col.insert_one(session_data)
        return session_id

    def get_user_chat_sessions(self, username):
        sessions = list(self.sessions_col.find({"username": username}))
                # Filter sessions for the user
        user_sessions = [
            {
                "session_id": session['session_id'],
                "session_name": session['session_name'],
                "created_at": session['created_at']
            } for session in sessions
        ]
        # Sort by creation time, most recent first
        return sorted(user_sessions, key=lambda x: x['created_at'], reverse=True)

    def save_chat_message(self, session_id, sender, message):
        message_data = {
            'session_id': session_id,
            'username': sender,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.messages_col.insert_one(message_data)

    def get_chat_messages(self, session_id):
        messages = list(self.messages_col.find({"session_id": session_id},{"_id": 0, "username": 1, "message": 1, "timestamp": 1}))
        return messages

    def delete_chat_session(self, session_id):
        # Delete session and associated messages
        self.sessions_col.delete_one({"session_id": session_id})
        self.messages_col.delete_many({"session_id": session_id})
        return True