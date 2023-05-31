import streamlit as st
import requests
import json
from PIL import Image
from flask import Flask, jsonify, request
import subprocess
import base64
#from PIL import Image
import io

# Define the path to the Flask app.py file
flask_app_path = 'app.py'  # Replace with the actual path to your app.py file

# Run the Flask app.py file using subprocess
subprocess.run(['python', flask_app_path])

print("Flask app run successfully ....")

# Convert PIL image to base64
def pil_image_to_base64(pil_image):
    # Create an in-memory binary stream
    image_stream = io.BytesIO()

    # Save the PIL image to the stream in PNG format
    pil_image.save(image_stream, format='PNG')

    # Retrieve the contents of the stream
    image_bytes = image_stream.getvalue()

    # Encode the image bytes as base64
    base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    return base64_encoded_image
# from flask import Flask, jsonify, request
# import requests
# import json

# #######################################
# img_path = "_G6A7982.jpg"

# url = "http://127.0.0.1:5000"

# headers = {"Content-Type":"application/json"}

# data = {"image":img_path}

# print("here")
# res = requests.post(url, headers=headers, data=json.dumps(data))
# print(res.status_code)
# print(res.json())

def main():
    st.title("Flask and Streamlit Integration")
    #img_path = st.text_input("Enter some input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    headers = {"Content-Type":"application/json"}
    if st.button("Submit"):
        image = Image.open(uploaded_file)
        img_base64 = pil_image_to_base64(pil_image=image)
        data = {"image":img_base64}
        print("here")
        res = requests.post("http://127.0.0.1:5000", headers=headers, data=json.dumps(data))
        print(res.status_code)
        print(res.json())
        st.write("Flask app response:", res.json())

if __name__ == "__main__":
    main()
