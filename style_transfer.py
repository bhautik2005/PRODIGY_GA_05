import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import copy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
def load_image(img_path, size=(512, 512)):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Image saver
def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(path)

# Gram Matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Normalization
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Load images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
input_img = content_img.clone()

# VGG19
cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Mean/Std for normalization
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Layers to extract features
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Build model
content_losses = []
style_losses = []

model = nn.Sequential(Normalization(cnn_normalization_mean, cnn_normalization_std)).to(device)

i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f'conv_{i}'
    elif isinstance(layer, nn.ReLU):
        name = f'relu_{i}'
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f'pool_{i}'
    elif isinstance(layer, nn.BatchNorm2d):
        name = f'bn_{i}'
    else:
        continue

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target = model(style_img).detach()
        style_loss = StyleLoss(target)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# Trim model after last content/style loss
for j in range(len(model) - 1, -1, -1):
    if isinstance(model[j], (ContentLoss, StyleLoss)):
        break
model = model[:j+1]

# Optimize
input_img.requires_grad_(True)
optimizer = optim.LBFGS([input_img])

# Training
num_steps = 300
style_weight = 1e6
content_weight = 1
run = [0]

print("Optimizing...")

while run[0] <= num_steps:
    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        loss = style_weight * style_score + content_weight * content_score
        loss.backward()

        if run[0] % 50 == 0:
            print(f"Step {run[0]}:")
            print(f"Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

        run[0] += 1
        return loss

    optimizer.step(closure)

# Save final output
with torch.no_grad():
    input_img.clamp_(0, 1)
save_image(input_img, "output.jpg")
print("Neural Style Transfer complete! Output saved as output.jpg")
# ✅ Optional: Display the image using matplotlib
import matplotlib.pyplot as plt
plt.imshow(transforms.ToPILImage()(input_img.squeeze(0).cpu()))
plt.title("Stylized Image")
plt.axis("off")
plt.show()