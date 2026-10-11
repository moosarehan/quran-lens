"""Surah names come from the complete Tanzil catalog, not an example list."""
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import get_close_matches
from .config import DATA

def normalize(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    text = re.sub(r"^(?:al|ash|as|an|at|ad|az|ar)[\s-]+", "", text)
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^\w]+", "", text)
    return re.sub(r"([aeiou])\1+", r"\1", text)

class SurahCatalog:
    def __init__(self):
        self.chapters = [dict(s.attrib) for s in ET.parse(DATA / "quran-data.xml").getroot().findall(".//suras/sura")]
        self.aliases = {}
        for chapter in self.chapters:
            for field in ("name", "tname", "ename"):
                name = chapter[field]
                self.aliases.setdefault(normalize(name), set()).add(int(chapter["index"]))
                self.aliases.setdefault(normalize(re.sub(r"^(?:Al|Ash|As|An|At|Ad|Az|Ar)-", "", name)), set()).add(int(chapter["index"]))

    def resolve(self, name):
        if str(name).isdigit():
            number = int(name)
            return number if 1 <= number <= len(self.chapters) else None
        normalized = normalize(name)
        if normalized in self.aliases:
            choices = self.aliases[normalized]
            return next(iter(choices)) if len(choices) == 1 else None
        matches = get_close_matches(normalized, self.aliases, n=2, cutoff=0.72)
        choices = set().union(*(self.aliases[m] for m in matches))
        return next(iter(choices)) if len(choices) == 1 else None

    def resolve_prefix(self, text):
        """Recognize metadata names before trailing query phrases."""
        words = text.split()
        for end in range(len(words), 0, -1):
            candidate = " ".join(words[:end])
            if normalize(candidate) in self.aliases or candidate.isdigit():
                return self.resolve(candidate)
        return self.resolve(text)

    def prompt_catalog(self):
        return "\n".join(f"{c['index']}: {c['tname']} / {c['ename']} / {c['name']} ({c['ayas']} ayahs)" for c in self.chapters)
