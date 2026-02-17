import faiss
import streamlit as st
import torch
import torch.nn as nn
import os

from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import models, transforms

# ------------------------
# DEVICE
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# PATHS
# ------------------------
MODEL_PATH = "best_model.pth"
EMB_PATH = "embeddings.pt"

# ------------------------
st.title("Monkey Classifier")

uploaded_file = st.file_uploader("Загрузить картинку", type = ["jpg", "png", "gpeg"])

# ------------------------
# MODEL
# ------------------------
model = models.resnet18(pretrained = False)
model.fc = nn.Linear(512, 6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

#модель_для_эмбеддингов
emb_model = models.resnet18(pretrained=False)
emb_model.fc = nn.Identity()
emb_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict = False)
emb_model = emb_model.to(device)
emb_model.eval()

# ------------------------
# TRANSFORM
# ------------------------
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# ------------------------
# LOAD EMBEDDINGS
# ------------------------

# Загружаем_заранее_сохранённые_embeddings
all_embeddings, image_path = torch.load(
    EMB_PATH,
    map_location=torch.device("cpu")
)

#FAISS
#создаем_индекс
#тензор -> numpy
emb_np = all_embeddings.numpy().astype("float32")
#нормализируем
faiss.normalize_L2(emb_np)
#создаем_место_для_хранения
index = faiss.IndexFlatIP(512)
#Добавляем_ембеддинги_в_индекс
index.add(emb_np)

def find(emb_new, image_path):
    query = emb_new.numpy().astype("float32")

    faiss.normalize_L2(query)

    D, I = index.search(query, 5)

    top5_paths = [image_path[i] for i in I[0]]
    return top5_paths


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    st.image(image, caption = "Загруженная картинка", width = 400)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_name = ["Ateles_geoffroyi", "Cebus_imitator", "Gorilla_beringei",
               "Macaca_fascicularis", "Pan_troglodytes", "Pongo_pygmaeus"]

    predicted_class = class_name[predicted.item()]

    st.subheader(f"Предсказанный класс: {predicted_class}")

    with torch.no_grad():
        emb_new = emb_model(image_tensor).cpu()
        top5 = find(emb_new, image_path)

    st.subheader("5 похожих изображений")

    cols = st.columns(5)

    for col, path, in zip(cols, top5):
        col.image(path, use_container_width=True)










