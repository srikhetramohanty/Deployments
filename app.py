####################################################################################


from flask import Flask, jsonify, request, render_template
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
print('Model loaded....')

# define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((640,640)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # get the uploaded image from the form
        img = request.files['image']
        #print("reached here")
        #img = request.get_json()["image"]
        #print(img)
        #open and transform the image
        img = Image.open(img)
        img_tensor = transform(img)

        #add batch dimension to the image
        img_tensor = img_tensor.unsqueeze(0)

        ##########################################


        # Define path to the image file
        source = img_tensor
        #print(img)
        #print(source)

        # Run inference on the source
        results = model(source)  # list of Results objects
        ##########################################

        # # pass the image through the model
        # with torch.no_grad():
        #     output = model(img_tensor)

        # # get the predicted class label
        # _, predicted = torch.max(output.data, 1)
        # label = predicted.item()
        
        #return jsonify({'class_label': label})
        return results[0].tojson()

    # return a JSON response with the predicted class label
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000",debug=True)
