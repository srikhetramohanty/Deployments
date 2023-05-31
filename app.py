####################################################################################


from flask import Flask, jsonify, request, render_template
from torchvision import models, transforms
from PIL import Image
import torch
import base64
from PIL import Image
import io

app = Flask(__name__)

# load the pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Convert base64 to PIL image
def base64_to_pil_image(base64_image):
    # Decode the base64 image to bytes
    image_bytes = base64.b64decode(base64_image)

    # Create a BytesIO stream from the image bytes
    image_stream = io.BytesIO(image_bytes)

    # Open the image stream using PIL
    pil_image = Image.open(image_stream)

    return pil_image

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # get the uploaded image from the form
        ##img = request.files['image']
        #print("reached here")
        img = request.get_json()["image"]
        #print(img)
        # open and transform the image
        #img = Image.open(img)
        img = base64_to_pil_image(base64_image=img)
        img_tensor = transform(img)

        # add batch dimension to the image
        img_tensor = img_tensor.unsqueeze(0)

        # pass the image through the model
        with torch.no_grad():
            output = model(img_tensor)

        # get the predicted class label
        _, predicted = torch.max(output.data, 1)
        label = predicted.item()
        
        return jsonify({'class_label': label})

    # return a JSON response with the predicted class label
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000",debug=True)
