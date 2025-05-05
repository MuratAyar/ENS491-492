# firebase/firebase_init.py
import firebase_admin
from firebase_admin import credentials, firestore, storage

import os

# Dosya yolu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

# Uygulama zaten başlatılmış mı kontrol et
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': '<YOUR_BUCKET_NAME>.appspot.com'  # ← BU KISMI AŞAĞIDA AÇIKLAYACAĞIM
    })

# Firestore & Storage clients
db = firestore.client()
bucket = storage.bucket()
