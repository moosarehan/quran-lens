import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

TRANSLATION = "Saheeh International"

def parse_verses(path: Path) -> dict[str, str]:
    verses = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"(\d+)\|(\d+)\|(.+)", line)
        if not match:
            raise ValueError(f"Unexpected verse format in {path.name}")
        chapter, ayah, text = match.groups()
        key = f"{int(chapter)}:{int(ayah)}"
        if key in verses:
            raise ValueError(f"Duplicate verse {key}")
        verses[key] = text  # Preserve the source text, including whitespace.
    return verses

class TanzilLoader(BaseLoader):
    """Load and align numbered Tanzil files by verse key, never by position."""
    def __init__(self, directory: Path):
        self.directory = directory

    def lazy_load(self):
        arabic = parse_verses(self.directory / "quran-uthmani.txt")
        english = parse_verses(self.directory / "en.sahih.txt")
        surahs = ET.parse(self.directory / "quran-data.xml").getroot().findall(".//suras/sura")
        expected = {f"{s.attrib['index']}:{a}" for s in surahs for a in range(1, int(s.attrib['ayas']) + 1)}
        if len(surahs) != 114 or len(expected) != 6236:
            raise ValueError("Unexpected Tanzil chapter metadata")
        if set(arabic) != expected or set(english) != expected:
            raise ValueError("Arabic/English verse keys do not match all 6,236 expected verses")
        for surah in surahs:
            number = int(surah.attrib["index"])
            for ayah in range(1, int(surah.attrib["ayas"]) + 1):
                key = f"{number}:{ayah}"
                text = arabic[key]
                header = None
                # Tanzil can prepend an unnumbered opening formula to ayah 1.
                # Separate it, preserving both pieces verbatim and the raw file.
                opening = arabic['1:1']
                parts = text.split(' ', len(opening.split()))
                normalize = lambda value: ''.join(c for c in unicodedata.normalize('NFKC', value)
                                                 if not unicodedata.category(c).startswith('M'))
                if ayah == 1 and len(parts) > len(opening.split()) and normalize(' '.join(parts[:-1])) == normalize(opening):
                    header = {'arabic_text': ' '.join(parts[:-1]), 'english_translation': english['1:1']}
                    text = parts[-1]
                yield Document(page_content=f"{text}\n{english[key]}", metadata={
                    "verse_key": key, "surah_number": number, "ayah_number": ayah,
                    "surah_name": surah.attrib["tname"], "surah_english_name": surah.attrib["ename"],
                    "surah_arabic_name": surah.attrib["name"],
                    "arabic_text": text, "english_translation": english[key],
                    "opening_header": header,
                    "translation_name": TRANSLATION, "translation_id": "en.sahih",
                    "document_type": "ayah", "source": "Tanzil Project",
                    "source_url": f"https://tanzil.net/#trans/en.sahih/{key}",
                })

