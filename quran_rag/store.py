import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from .config import VERSE_DB, DATA
from .morphology import MORPHOLOGY_FILE, install, unprefixed_words, proper_name_candidates

def tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)

def normalize_arabic(text):
    """Normalize search copies only; canonical Uthmani text stays untouched."""
    text = unicodedata.normalize("NFKC", text).replace("\u0671", "\u0627")
    return "".join(c for c in text if not unicodedata.category(c).startswith("M")
                   and c not in "\u0640\u06e5\u06e6").replace("ءا", "آ")

def is_arabic_letter(char):
    return unicodedata.category(char).startswith("L") and "ARABIC" in unicodedata.name(char, "")

def arabic_word(word):
    normalized = normalize_arabic(word).strip()
    if not normalized or not all(is_arabic_letter(c) for c in normalized):
        raise ValueError("Provide one exact Arabic word to count in the Quran's Arabic text.")
    return normalized

def arabic_tokens(text):
    return [word for word in re.findall(r"[^\W\d_]+", normalize_arabic(text), re.UNICODE)
            if all(is_arabic_letter(c) for c in word)]


def arabic_search_words(text):
    """Both Uthmani defective and expanded alif spellings; no root expansion."""
    result = []
    for word in text.split():
        variants = {normalize_arabic(word).replace('ى', 'ي'),
                    normalize_arabic(word.replace('ٰ', 'ا')).replace('ى', 'ي')}
        variants = {''.join(c for c in value if is_arabic_letter(c)) for value in variants}
        variants.discard('')
        if variants:
            result.append(variants)
    return result


def contains_phrase(haystack, needle):
    return any(all(needle[j] & haystack[start+j] for j in range(len(needle)))
               for start in range(len(haystack)-len(needle)+1))

class VerseStore:
    def __init__(self, path=VERSE_DB, morphology_path=MORPHOLOGY_FILE):
        self.path = path
        self.morphology_path = morphology_path
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS verses (verse_key TEXT PRIMARY KEY, surah_number INTEGER NOT NULL, ayah_number INTEGER NOT NULL, metadata TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS chapter_ayah ON verses(surah_number, ayah_number)")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ingest(self, documents):
        rows = [(d.metadata["verse_key"], d.metadata["surah_number"], d.metadata["ayah_number"], json.dumps(d.metadata, ensure_ascii=False)) for d in documents]
        with self.connect() as conn:
            conn.execute("DELETE FROM verses")
            conn.executemany("INSERT INTO verses VALUES (?, ?, ?, ?)", rows)

    def all(self):
        with self.connect() as conn:
            return [json.loads(r[0]) for r in conn.execute("SELECT metadata FROM verses ORDER BY surah_number, ayah_number")]

    def get(self, key):
        with self.connect() as conn:
            row = conn.execute("SELECT metadata FROM verses WHERE verse_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def chapter(self, number, start=1, end=999):
        with self.connect() as conn:
            return [json.loads(r[0]) for r in conn.execute("SELECT metadata FROM verses WHERE surah_number=? AND ayah_number BETWEEN ? AND ? ORDER BY ayah_number", (number, start, end))]

    def word_search(self, word):
        needle = tokens(word)
        if len(needle) != 1:
            raise ValueError("Provide a single English word")
        return [v for v in self.all() if needle[0] in tokens(v["english_translation"])]

    def literal_search(self, expression, language='english', match='word'):
        if language == 'arabic' and match == 'lemma':
            matches = self.arabic_word_count(expression)['verses']
            # A transliterated name can be resolved to a modern dictionary
            # spelling while the morphology corpus records an older spelling.
            # If that lemma is absent, retain exact Arabic word matching rather
            # than silently returning no verses. This does not expand a root.
            return matches or self.literal_search(expression, 'arabic', 'word')
        if language == 'arabic':
            needle = arabic_search_words(expression)
            if not needle:
                raise ValueError('Provide an Arabic word or phrase')
            starts = unprefixed_words(str(self.morphology_path)) if len(needle) > 1 else {}
            result = []
            for v in self.all():
                words = arabic_search_words(v['arabic_text'])
                for start in range(len(words)-len(needle)+1):
                    initial = words[start]
                    annotated = starts.get((v['verse_key'], start+1))
                    if annotated:
                        initial = initial | set().union(*arabic_search_words(annotated))
                    if needle[0] & initial and all(needle[j] & words[start+j] for j in range(1, len(needle))):
                        result.append(v)
                        break
            return result
        if language != 'english':
            raise ValueError('Search language must be Arabic or English')
        needle = [{t} for t in tokens(expression)]
        if not needle:
            raise ValueError('Provide a word or phrase')
        return [v for v in self.all() if contains_phrase([{t} for t in tokens(v['english_translation'])], needle)]

    def header_search(self, expression):
        opening = self.get('1:1')
        needle = arabic_search_words(expression)
        if not opening or not needle or not contains_phrase(arabic_search_words(opening['arabic_text']), needle):
            return []
        return [{'surah_number': v['surah_number'], 'surah_name': v['surah_name'],
                 **v['opening_header'], 'document_type': 'unnumbered_header',
                 'source_url': f"https://quran.com/{v['surah_number']}"}
                for v in self.all() if v.get('opening_header')]

    def arabic_word_count(self, word, mode="lemma"):
        if mode not in ("lemma", "surface"):
            raise ValueError("Count mode must be lemma or surface")
        if mode == "lemma":
            return self.lemma_count(word)
        return self.surface_count(word)

    def lemma_count(self, word):
        needle = arabic_word(word)
        with self.connect() as conn:
            install(conn, self.morphology_path, normalize_arabic)
            # Exact lemma takes priority over ambiguous inflected stem forms.
            lemmas = [r[0] for r in conn.execute("SELECT DISTINCT lemma FROM morphology WHERE lemma=?", (needle,))]
            if not lemmas:
                lemmas = [r[0] for r in conn.execute("SELECT DISTINCT lemma FROM morphology WHERE form=? UNION SELECT lemma FROM morphology_aliases WHERE form=?", (needle, needle))]
            if not lemmas:
                lemmas = proper_name_candidates(str(self.morphology_path), needle, normalize_arabic)
            if len(lemmas) > 1:
                raise ValueError("Ambiguous Arabic form; specify a lemma: " + ", ".join(lemmas))
            lemma = lemmas[0] if lemmas else needle
            rows = conn.execute("SELECT v.metadata, COUNT(DISTINCT m.position) FROM morphology m JOIN verses v ON v.verse_key=m.verse_key WHERE m.lemma=? GROUP BY v.verse_key ORDER BY v.surah_number,v.ayah_number", (lemma,)).fetchall()
        return {"word": lemma, "normalized_word": lemma, "requested_word": word,
                "occurrences": sum(r[1] for r in rows), "verse_count": len(rows),
                "language": "arabic", "mode": "lemma", "source": "Quranic Arabic Corpus v0.4; Tanzil Uthmani Arabic",
                "source_url": "https://corpus.quran.com/", "method": "Arabic lemma occurrences, including attached articles, clitics and inflections; other lemmas of the same root are excluded. For stems without a corpus lemma, normalized stem spelling is used.",
                "verses": [json.loads(r[0]) for r in rows]}

    def surface_count(self, word):
        needle = arabic_word(word)
        verses, occurrences = [], 0
        for verse in self.all():
            count = arabic_tokens(verse["arabic_text"]).count(needle)
            if count:
                verses.append(verse)
                occurrences += count
        return {"word": word, "normalized_word": needle, "occurrences": occurrences,
                "verse_count": len(verses), "language": "arabic", "source": "Tanzil Uthmani Arabic", "mode": "surface",
                "method": "Whole written tokens; diacritics, tatweel and recitation marks ignored. Wasla normalized to alif; prefixes, suffixes and root variants remain separate.",
                "verses": verses}
