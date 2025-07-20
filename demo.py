import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.patches as patches
import pytesseract

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import mediapipe as mp

os.environ["CURL_CA_BUNDLE"] = ""

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = r"C:\Users\shrad\TrendWeave\TrendWeave\extracted_data\Catwalks__Women_s_Key_Items___Soft_Accessories_A_W_25_26_en"
OUTPUT_DIR = r"outputs18"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize models
yolo = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
pose_detector = mp.solutions.pose.Pose(static_image_mode=True)

FEATURE_PROMPTS_FEMALE = {
    "palette": ["pastel", "neon", "monochrome", "earth tones", "bold colors"],
    "dress_silhouette": ["bodycon dress", "fit and flare", "wrap dress", "maxi dress", "shirt dress"],
    "neckline": ["v-neck", "square neckline", "halter neck", "off-shoulder", "sweetheart"],
    "sleeve_type": ["sleeveless", "cap sleeves", "puffed sleeves", "bell sleeves", "long sleeves"],
    "top_silhouette": ["crop top", "peplum", "wrap top", "fitted blouse", "camisole"],
    "bottom_style": ["mini skirt", "culottes", "flared pants", "wide-leg pants", "denim shorts"],
    "outerwear": ["denim jacket", "blazer", "cardigan", "long coat", "shrug"],
    "shoe_type": ["heels", "ballet flats", "ankle boots", "strappy sandals", "platform sneakers"],
    "bag_type": ["tote bag", "clutch", "mini backpack", "crossbody bag", "shoulder bag"],
    "accessory": ["hoop earrings", "sunglasses", "hairband", "watch", "scarf"],
    "hairstyle": ["high bun", "loose curls", "braids", "straight hair", "ponytail"],
    "setting": ["beach casual", "party night", "brunch look", "street style", "office wear"],
    "fabric_type": ["satin", "chiffon", "linen", "cotton", "organza"],
    "season": ["summer", "winter", "spring", "fall"],
    "style_theme": ["boho", "y2k", "minimalist", "vintage", "elegant chic"]
}
FEATURE_PROMPTS_MALE = {
    "palette": ["monochrome", "earth tones", "muted tones", "navy and grey", "bold colors"],
    "dress_silhouette": [],
    "neckline": ["crew neck", "v-neck", "polo", "buttoned collar", "mandarin collar"],
    "sleeve_type": ["short sleeves", "rolled sleeves", "long sleeves", "sleeveless", "half sleeves"],
    "top_silhouette": ["oversized tee", "fitted t-shirt", "button-up shirt", "hoodie", "tank top"],
    "bottom_style": ["chinos", "joggers", "denim jeans", "cargo shorts", "tailored trousers"],
    "outerwear": ["leather jacket", "bomber jacket", "blazer", "hoodie", "denim jacket"],
    "shoe_type": ["sneakers", "loafers", "derby shoes", "slip-ons", "boots"],
    "bag_type": ["messenger bag", "backpack", "duffel bag", "crossbody bag"],
    "accessory": ["watch", "cap", "chain", "sunglasses", "bracelet"],
    "hairstyle": ["undercut", "slick back", "buzz cut", "curly top", "messy hair"],
    "setting": ["street casual", "gym wear", "date night", "beach look", "smart casual"],
    "fabric_type": ["denim", "cotton", "linen", "polyester", "wool"],
    "season": ["summer", "winter", "spring", "monsoon"],
    "style_theme": ["streetwear", "minimalist", "grunge", "classic", "sporty"]
}
FEATURE_PROMPTS_UNISEX = {
    "palette": ["black & white", "pastel", "bold primary colors", "earth tones", "neon"],
    "shoe_type": ["sneakers", "sandals", "slip-ons", "boots", "canvas shoes"],
    "bag_type": ["crossbody bag", "backpack", "tote bag", "belt bag", "messenger bag"],
    "accessory": ["watch", "sunglasses", "bracelet", "ring", "hat"],
    "setting": ["streetwear", "travel outfit", "festival wear", "college casual", "lounge"],
    "fabric_type": ["denim", "cotton", "linen", "nylon", "jersey"],
    "season": ["summer", "winter", "rainy", "fall"],
    "style_theme": ["minimalist", "y2k", "boho", "athleisure", "vintage"]
}


def detect_with_yolov8(image_path):
    results = yolo(image_path)
    detections = []
    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()
        detections.append({"label": label, "bbox": bbox, "confidence": conf})
    return detections

def crop_image(image_path, bbox):
    image = Image.open(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))

def clip_attributes(pil_img, feature_prompts):
    inputs = {}
    outputs = {}
    image_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    for category, labels in feature_prompts.items():
        if not labels:
            continue
        text_inputs = clip_processor(text=labels, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        best_idx = int(np.argmax(similarity))
        best_label = labels[best_idx]
        outputs[category] = {"label": best_label, "confidence": float(similarity[best_idx])}
    return outputs

def analyze_color_texture(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return dominant_colors.tolist(), lbp.flatten().tolist()

def detect_pose(image_path):
    image = cv2.imread(image_path)
    results = pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    return [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in results.pose_landmarks.landmark]

def detect_gender(pil_img):
    prompts = ["a man wearing modern clothes", "a woman in stylish clothes", "unisex clothing"]
    inputs = clip_processor(text=prompts, images=pil_img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy().flatten()
    return prompts[np.argmax(probs)], float(np.max(probs))

def process_image(image_path):
    rel_path = os.path.relpath(image_path, IMAGE_DIR)
    safe_name = rel_path.replace(os.sep, "__").replace("/", "__")
    json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(safe_name)[0]}_analysis.json")
    result = {"image": image_path, "items": []}
    yolo_detections = detect_with_yolov8(image_path)
    for det in yolo_detections:
        bbox = det["bbox"]
        cropped = crop_image(image_path, bbox)
        gender_label, _ = detect_gender(cropped)
        if "woman" in gender_label:
            prompt_set = FEATURE_PROMPTS_FEMALE
        elif "man" in gender_label:
            prompt_set = FEATURE_PROMPTS_MALE
        else:
            prompt_set = FEATURE_PROMPTS_UNISEX

        clip_output = clip_attributes(cropped, prompt_set)
        dominant_colors, texture_hist = analyze_color_texture(cropped)
        item = {
            "yolo_label": det["label"],
            "bbox": bbox,
            "confidence": det["confidence"],
            "clip_attributes": clip_output,
            "dominant_colors": dominant_colors,
            "texture_histogram": texture_hist
        }
        result["items"].append(item)
    pose_keypoints = detect_pose(image_path)
    result["pose_keypoints"] = pose_keypoints
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return result

def batch_process_images(image_dir):
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} images.")
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            process_image(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def display_summary(data):
    fig, ax = plt.subplots()
    img = Image.open(data["image"])
    ax.imshow(img)
    for item in data["items"]:
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, item["yolo_label"], color='lime', fontsize=12, weight='bold')

    summary_lines = []
    for item in data["items"]:
        for k, v in item["clip_attributes"].items():
            summary_lines.append(f"{k.title()}: {v['label']}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.gcf().text(0.72, 0.4, "\n".join(summary_lines), fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    batch_process_images(IMAGE_DIR)
    print("All images processed. Results saved to", OUTPUT_DIR)