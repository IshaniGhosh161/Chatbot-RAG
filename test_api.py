import requests
import json

# Base URL of the API
BASE_URL = 'http://localhost:5000'

# Helper function for pretty printing
def pretty_print(response):
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    print("\n")

# 1. User Registration
def test_register():
    register_data = {
        "username": "testuser",
        "password": "password123",
        "email": "testuser@example.com"
    }
    
    response = requests.post(f"{BASE_URL}/api/register", json=register_data)
    pretty_print(response)
    return response

# 2. User Login
def test_login():
    login_data = {
        "username": "testuser",
        "password": "password123"
    }
    
    response = requests.post(f"{BASE_URL}/api/login", json=login_data)
    pretty_print(response)
    return response

# 3. Create a Chat Session
def test_create_session():
    session_data = {
        "username": "testuser",
        "session_name": "Test Session"
    }
    
    response = requests.post(f"{BASE_URL}/api/sessions", json=session_data)
    pretty_print(response)
    return response

# 4. Get User Sessions
def test_get_sessions():
    response = requests.get(f"{BASE_URL}/api/sessions", params={"username": "testuser"})
    pretty_print(response)
    return response

# 5. Save a Message to a Session
def test_save_message(session_id,message_data):   
    response = requests.post(f"{BASE_URL}/api/sessions/{session_id}/messages", json=message_data)
    pretty_print(response)
    return response

# 6. Test Bot Response Generation
def test_generate_response(session_id,message_data):
    response = requests.post(f"{BASE_URL}/api/sessions/{session_id}/generate", json=message_data)
    pretty_print(response)
    return response

# 7. Get Messages from a Session
def test_get_messages(session_id):
    response = requests.get(f"{BASE_URL}/api/sessions/{session_id}/messages")
    pretty_print(response)
    return response

# 8. Delete a Session
def test_delete_session(session_id):
    response = requests.delete(f"{BASE_URL}/api/sessions/{session_id}")
    pretty_print(response)
    return response

# Main testing function
def run_all_tests():
    
    message_data = {
        "username": "testuser",
        "message": "Real estate agent"
    }
    # Register a user
    register_response = test_register()
    
    # Login
    login_response = test_login()
    
    # Create a session
    create_session_response = test_create_session()
    session_id = create_session_response.json().get('session_id')
    
    # Get sessions
    test_get_sessions()
    
    # Save a message
    test_save_message(session_id,message_data)
    
    # Test bot response generation
    test_generate_response(session_id,message_data)
    
    # Get messages
    test_get_messages(session_id)
    
    # Delete session
    test_delete_session(session_id)

# Run the tests
if __name__ == "__main__":
    run_all_tests()