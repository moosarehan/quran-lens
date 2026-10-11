import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", encoding="utf-8-sig")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
VERSE_DB = DATA / "quran.sqlite"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MODEL = os.getenv("EMBEDDING_MODEL", "@cf/baai/bge-m3")
EMBEDDING_ENGINE = "cloudflare-dense-v1"
COLLECTION = "quran_ayahs_cloudflare_" + hashlib.sha256((MODEL + EMBEDDING_ENGINE).encode()).hexdigest()[:12]
ARABIC_COLLECTION = COLLECTION + "_arabic"
VECTOR_COLLECTIONS = {"english": COLLECTION, "arabic": ARABIC_COLLECTION}
TEXT_FIELDS = {"english": "english_translation", "arabic": "arabic_text"}
