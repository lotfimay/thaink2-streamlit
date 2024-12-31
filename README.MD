# Thaink² Technical Test Application

This application connects to a database, generates forecasts using multiple models, and provides visualizations for comparison. The application is built with Python, Streamlit, Plotly, and uses the Thaink² API for forecasting.

---

## Requirements

- Python version **≥ 3.10.0** is required.

---

## How to Run?

Follow these steps to set up and run the application:

### 1. Unzip the Folder
Unzip the provided folder to your local machine and navigate to the project directory:
```bash
unzip project_folder.zip
cd project_folder
```

### 2. Create a Virtual Environment
Set up a Python virtual environment to keep dependencies isolated:
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment
Activate the virtual environment using the appropriate command for your operating system:

- **On Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **On macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
Install the required Python dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Run the `app.py` file using Streamlit:
```bash
streamlit run app.py
```

### 6. Access the Application
Once the application is running, open your browser and navigate to:
```plaintext
http://localhost:8501
```

---
