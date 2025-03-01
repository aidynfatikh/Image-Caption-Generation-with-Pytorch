import streamlit as st
import model as models
from PIL import Image
import pickle

import torchvision.transforms as transforms
import torch

# HYPERPARAMETERS (MAKE SURE THEY ARE SAME AS IN TRAIN.PY)
batch_size = 32
hidden_size = 256
num_layers = 2
n_embd = 196
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LOAD VOCABULARY
with open("dataset/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab.stoi)
transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


encoder = models.EncoderCNN(n_embd).to(device)
decoder = models.DecoderRNN(n_embd, hidden_size, vocab_size, num_layers).to(device)

if True: # MAKE SURE YOU HAVE A TRAINED MODEL IN "checkpoints"
    epoch = 10 # WHICH EPOCH TO LOAD
    encoder.load_state_dict(torch.load(f"checkpoints/encoder_{epoch}.pth", weights_only=False))
    decoder.load_state_dict(torch.load(f"checkpoints/decoder_{epoch}.pth", weights_only=False))

encoder.eval()
decoder.eval()

st.title("Image Captioner")
st.write("Upload an image to generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ensure session state is initialized
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
        st.session_state.caption = ""

    # check if a new image is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file.name  
        st.session_state.caption = ""  # reset caption

    # generate caption if its empty
    if not st.session_state.caption:
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            features = encoder(image_tensor)
            output = decoder.sample(features, vocab)
            caption = " ".join(output)

        st.session_state.caption = caption  

    st.subheader("📝 Generated Caption:")
    st.write(st.session_state.caption)
