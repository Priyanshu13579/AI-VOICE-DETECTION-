import os                                   # Provides operating system utilities like file and directory handling
from flask import Flask, render_template, request, redirect, url_for, flash
# Imports Flask core class and helper functions for rendering templates, handling requests, redirects, URLs, and messages

from werkzeug.utils import secure_filename  # Ensures uploaded filenames are safe to store on the server
from predict import analyze_audio           # Imports the audio analysis (prediction) function

app = Flask(__name__)                       # Creates a Flask application instance

UPLOAD_FOLDER = 'uploads'                  # Defines the folder where uploaded files will be stored
if not os.path.exists(UPLOAD_FOLDER):       # Checks if the upload folder already exists
    os.makedirs(UPLOAD_FOLDER)              # Creates the upload folder if it does not exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Stores upload folder path in Flask configuration

ALLOWED_EXTENSIONS = {'wav'}                # Defines allowed file extensions for upload

def allowed_file(filename):                 # Function to check if uploaded file has valid extension
    """Check if the file has an allowed extension."""  # Documentation string
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    # Returns True if file has an extension and it is .wav

@app.route('/')                             # Maps the root URL to the index function
def index():
    return render_template('index.html')    # Renders and returns the main HTML upload page

@app.route('/result', methods=['POST'])     # Defines route to handle file upload form submission
def process():
    try:
        if 'file' not in request.files:     # Checks if file is present in the request
            flash("No file part in the request.")  # Shows error message to user
            return redirect(url_for('index'))      # Redirects back to home page

        file = request.files['file']        # Retrieves the uploaded file object

        if file.filename == '':             # Checks if user submitted without selecting a file
            flash("No file selected.")      # Shows error message
            return redirect(url_for('index'))  # Redirects back to home page

        if file and allowed_file(file.filename):  # Validates file existence and extension
            filename = secure_filename(file.filename)  # Sanitizes filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Builds full path to save uploaded file
            
            file.save(file_path)             # Saves uploaded file to uploads folder

            result = analyze_audio(file_path)  # Calls ML model to analyze the audio file

            return render_template('index.html', result=result)
            # Renders index page again and displays prediction result

        else:
            flash("Invalid file type. Only .wav files are allowed.")  # Error for wrong file type
            return redirect(url_for('index'))  # Redirects back to home page

    except Exception as e:
        print(f"Error processing file: {e}")  # Prints error details to console
        flash("An error occurred while processing the file.")  # Displays generic error to user
        return redirect(url_for('index'))     # Redirects back to home page

@app.errorhandler(404)                        # Handles 404 (page not found) errors
def page_not_found(e):
    return render_template('404.html'), 404  # Renders custom 404 error page

if __name__ == '__main__':                   # Ensures code runs only when file is executed directly
    app.run(debug=True)                      # Starts Flask development server with debug mode enabled
