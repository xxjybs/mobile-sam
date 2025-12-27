import torch
from models.mobile_sam_adapter import Mobile_sam_adapter

# Create model instance with input size 256
model = Mobile_sam_adapter(inp_size=256)

# Create test input of size 256x256
test_input = torch.randn(1, 3, 256, 256)

# Print model information
print("Model input size:", test_input.shape)
print("TinyViT input size:", model.image_encoder.img_size)
print("Feature map size:", model.image_embedding_size)

# Forward pass
with torch.no_grad():
    output = model(test_input)

# Print output information
print("Model output size:", output.shape)
print("Is output size 256x256:", output.shape[-2:] == (256, 256))
