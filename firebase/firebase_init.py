# firebase/firebase_init.py
import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv
load_dotenv()

import os

cred_path = os.getenv("FIREBASE_CREDENTIALS")
bucket_name = os.getenv("FIREBASE_BUCKET")

# Uygulama zaten başlatılmış mı kontrol et
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name  # ← BU KISMI AŞAĞIDA AÇIKLAYACAĞIM
    })

# Firestore & Storage clients
db = firestore.client()
bucket = storage.bucket()
