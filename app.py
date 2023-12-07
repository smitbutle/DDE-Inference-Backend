from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
import pandas as pd
import warnings
from inference import batch_process
app = Flask(__name__)

# Disable all warnings
warnings.filterwarnings("ignore")

# Set the upload folder
app.config['UPLOAD_FOLDER'] = './input_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_images():

    input_folder = "./input_images"
    output_folder = "./output_images"
    batch_process(input_folder, output_folder)
    return 'data.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    files = request.files.getlist('file')

    if not files:
        return "No selected file"

    # Check if the user submitted at least one valid image file
    valid_files = [file for file in files if file and allowed_file(file.filename)]
    if not valid_files:
        return "No valid image file selected"

    # Save the uploaded files to the UPLOAD_FOLDER
    uploaded_filenames = []
    for file in valid_files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploaded_filenames.append(filename)

    # Process the images and get the CSV file
    csv_filename = process_images()
    # Return a link to download the processed CSV file
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],"..", csv_filename),
                     as_attachment=True,
                     download_name='output.csv')

if __name__ == '__main__':
    app.run(debug=True)
