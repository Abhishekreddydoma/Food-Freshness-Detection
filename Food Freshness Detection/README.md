# Food Freshness Detection

A simple web application that lets you upload a food image to check its freshness.

---

## Setup & Run (Windows)

### 1. Create a virtual environment

```cmd
python -m venv venv
```

### 2. Activate the virtual environment

```cmd
venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Start the Flask server

```cmd
python app.py
```

### 5. Open in your browser

Navigate to:

```
http://127.0.0.1:5000
```

---

## Project Structure

```
Food_Freshness_Detection/
├── app.py              # Flask application entry point
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/
│   └── index.html      # Main HTML page
└── static/
    └── css/
        └── style.css   # Stylesheet
```

---

## Notes

- Accepted image formats: **JPG, JPEG, PNG**
- This is Stage 1 — the upload form is present but prediction is not yet implemented.
