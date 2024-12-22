#### Create Virtual Environment
python -m venv .venv

#### Activate the Virtual environment
.venv\Scripts\activate

#### Import required library
pip install -r requirements.txt

#### Run Flask app first
python api.py

#### Run streamlit app
streamlit run app.py --server.port=8502