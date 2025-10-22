# 🧠 Face Recognition System (ArcFace + SCRFD)

This project implements a **high-accuracy face recognition system** using the **InsightFace** library with **ArcFace embeddings** and **SCRFD detection**.  
It allows you to create a dataset, generate facial embeddings, and recognize users through images or webcam input.

---

## 📁 Folder Structure

dataset/

├── Thulasizwe/

├── 20251017_185806.jpg. 

 ├── 20251017_185806(0).jpg

│ ├── 20251017_185807.jpg

│ └── 20251017_185808.jpg

test/
├── image.png

└── test.jpeg

capture.py

recognize.py

reqN.py

Script.py

requirements.txt

<img width="904" height="332" alt="image" src="https://github.com/user-attachments/assets/d2a41008-5e9d-4f87-94cf-31dd382b36bf" />

<img width="1600" height="844" alt="image" src="https://github.com/user-attachments/assets/738f4989-36de-47b1-955c-ff5b015e074b" />

for script.py is allows you to integrate multiple cameras(multi-threading),Each camera runs in its own separate thread &Prevents blocking - if one camera freezes, others continue


> 💡 **Tip:** Take at least **15 photos per user** for reliable recognition.  
> One image can only provide around **30% accuracy**.

---

## ⚙️ Installation

Install all dependencies:

```bash
pip install -r requirements.txt

🧩 On the first run, InsightFace will automatically download pre-trained models
(stored in ~/.insightface/models).


The system uses:

ArcFace for embeddings

SCRFD for detection — faster and more accurate than MTCNN under varied lighting and angles.



🚀 How to Use

🔹 Step 1 — Generate Embeddings

Create the embeddings.pkl file from your dataset:

python recognize.py --generate

or

python reqN.py --generate

Assumes one face per image (if multiple faces exist, it processes the first).



🔹 Step 2 — Test Recognition on Images

Run recognition for test images:

python recognize.py --image test/test.jpeg


or

python reqN.py --image test/image.png


🔹 Step 3 — Real-Time Recognition via Webcam

Start webcam-based recognition:

python recognize.py --webcam


or

python reqN.py --webcam

Press q to exit webcam mode.


Multiply cameras:
python script.py --multicam

🧾 Script Overview

recognize.py	Detects both known and unknown users

reqN.py	Detects known users only










