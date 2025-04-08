# from flask import Flask, request
# from flask_cors import CORS
# import subprocess

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/run_file/<file_name>', methods=['POST'])
# def run_file(file_name):
#     try:
#         result = subprocess.run(["python", file_name], capture_output=True, text=True, check=True)
#         return {"output": result.stdout}, 200
#     except subprocess.CalledProcessError as e:
#         return {"error": e.stderr}, 500
#     except Exception as e:
#         return {"error": str(e)}, 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for browser compatibility

@app.route('/run_file/<file_name>', methods=['POST'])
def run_file(file_name):
    try:
        print(f"Executing: {file_name}")  # Log the file being executed
        result = subprocess.run(
            ["python", file_name], capture_output=True, text=True, check=True
        )
        print(f"Output: {result.stdout}")  # Print the script's output
        return {"output": result.stdout}, 200
    except subprocess.CalledProcessError as e:
        print(f"Error in script execution: {e.stderr}")  # Log errors
        return {"error": e.stderr}, 500
    except Exception as e:
        print(f"General error: {str(e)}")  # Log general exceptions
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
