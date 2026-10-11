"""Download unmodified source files with SHA-256 provenance."""
import hashlib
import json
from datetime import datetime, timezone
import httpx
from .config import DATA
from .loader import TanzilLoader

URLS = {
    "quran-uthmani.txt": "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true&marks=true&sajdah=true&tatweel=true",
    "en.sahih.txt": "https://tanzil.net/trans/?transID=en.sahih&type=txt-2",
    "quran-data.xml": "https://tanzil.net/res/text/metadata/quran-data.xml",
}

def main():
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for name, url in URLS.items():
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            if b"<html" in content[:500].lower():
                raise ValueError(f"Download returned HTML instead of data: {name}")
            (DATA / name).write_bytes(content)
            manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    documents = TanzilLoader(DATA).load()
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded and validated {len(documents)} aligned ayahs")

if __name__ == "__main__":
    main()
