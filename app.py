from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

# Tạo Flask app
app = Flask(__name__)

# Định nghĩa lại mô hình của bạn
class CheXNet(nn.Module):
    def __init__(self):
        super(CheXNet, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        return x

# Khởi tạo mô hình
model = CheXNet()

# Tải state_dict từ tệp đã lưu
model.load_state_dict(torch.load('D:/Semester 4/FlaskProject/chexnet_model.pth', map_location=torch.device('cpu')))
model.eval()

# Danh sách các lớp bệnh lý (tiếng Việt)
class_names = [
    'Xẹp phổi', 'Tim to', 'Tràn dịch', 'Thâm nhiễm', 'Khối u', 'Nốt',
    'Viêm phổi', 'Tràn khí màng phổi', 'Đông đặc', 'Phù nề', 'Khí phế thũng', 'Xơ hóa',
    'Dày màng phổi', 'Thoát vị'
]

# Hàm tiền xử lý ảnh
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Thêm batch dimension
    return image

# Hàm dự đoán
def get_prediction(image_bytes):
    input_image = preprocess_image(image_bytes)
    with torch.no_grad():
        output = model(input_image)
        predicted_labels = output[0]
    return {class_names[i]: float(predicted_labels[i]) for i in range(len(class_names))}

# Route để nhận và xử lý ảnh
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        image_bytes = file.read()
        prediction = get_prediction(image_bytes)
        return jsonify({'predictions': prediction})
    return jsonify({'error': 'Failed to process image'})

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
