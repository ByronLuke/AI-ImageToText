from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Initialize the processor and model for VQA (Visual Question Answering)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load your image
img_path = 'cyber_cat.jpg'  # Use the correct path to your image
raw_image = Image.open(img_path).convert('RGB')

# Ask a question related to the image
question = "What animal is this?"
inputs = processor(raw_image, question, return_tensors="pt")

# Generate the answer
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Question:", question)
print("Answer:", answer)
