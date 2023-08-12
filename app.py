####################################################################################


from flask import Flask, jsonify, request, render_template,send_file
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import json
from PIL import Image, ImageDraw, ImageFont
import io
import base64


app = Flask(__name__)

def image_to_base64(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)
    return base64.b64encode(img_byte_array.read()).decode()

def draw_bboxes_on_image(image, bbox_predictions):
    # Create a copy of the input image to avoid modifying the original image
    output_image = image.copy()

    # Initialize the drawing context
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.load_default()  # Load a default font for text

    # COCO class names and associated colors
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        # ... and so on
    ]

    # Generate a list of distinct colors for classes
    colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (0, 128, 255),   # Sky Blue
        (255, 0, 128),   # Pink
        (0, 255, 128),   # Lime Green
        (128, 255, 0),   # Spring Green
        (0, 128, 128),   # Teal
        (128, 128, 0),   # Olive
        (192, 192, 192), # Silver
    ]

    # Create a dictionary to map each COCO class to a color
    class_color_map = {coco_class: color for coco_class, color in zip(coco_classes, colors)}

    for bbox_info in bbox_predictions:
        class_name = bbox_info["name"]
        confidence = bbox_info["confidence"]
        box = bbox_info["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        bbox_coordinates = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

        # Get the color for the class from the class-color mapping
        color = class_color_map.get(class_name, (0, 0, 0))  # Default to black for unknown classes

        # Draw bounding box
        draw.line(bbox_coordinates, fill=color, width=2)

        # Construct text to display
        text = f'{class_name} ({confidence:.2f})'

        # Calculate text size and position
        text_size = draw.textsize(text, font=font)
        text_position = (x1, y1 - text_size[1] - 5)

        # Draw text
        draw.text(text_position, text, fill=color, font=font)

    return output_image


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
        #return results[0].tojson()
        ###########################################################################################
        # Convert JSON string to dictionary
        results_dictionary = json.loads(results[0].tojson())
        # Load an example PIL image
        #image_path = "images.jpg"
        #image = Image.open(image_path)

        # Draw bounding boxes on the image
        output_image = draw_bboxes_on_image(img, results_dictionary)
        
        # Convert PIL image to bytes
        #img_byte_array = io.BytesIO()
        #output_image.save(img_byte_array, format='JPEG')  # Use 'PNG' if you prefer PNG format
        #img_byte_array.seek(0)
        
        #img2_byte_array = io.BytesIO()
        #img.save(img2_byte_array, format='JPEG')
        #img2_byte_array.seek(0)
        # Convert images to base64-encoded strings
        img1_base64 = image_to_base64(img)
        img2_base64 = image_to_base64(output_image)
        
        images_list = [img1_base64,img2_base64]
        description_list = ["Original Image","Inferenced Image"]
        combination = zip(images_list,description_list)
        ###########################################################################################
        
        #return send_file(img_byte_array, mimetype='image/jpeg')
        return render_template('images_with_text.html', images_and_descriptions=combination)

    # return a JSON response with the predicted class label
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000",debug=True)
