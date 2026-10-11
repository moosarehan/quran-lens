"""QAC annotations, preserved verbatim on disk; derived search rows only."""
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from .config import DATA

MORPHOLOGY_FILE = DATA / 'morphology' / 'quranic-corpus-morphology-0.4.txt'
BW = {
    **dict(zip("'|>&<}AbptvjHxd*rzs$SDTZEgfqklmnhwYy", "ءآأؤإئا بةتثجحخدذرزسشصضطظعغفقكلمنهوىي".replace(' ', ''))),
    **dict(zip('FNKaui~o`{', 'ًٌٍَُِّْٰٱ')),
}

def arabic(value):
    # Other QAC symbols annotate recitation, not lexical letters.
    return ''.join(BW[c] for c in value if c in BW)


def root_key(root):
    return ''.join('ء' if c in 'اأإآؤئ' else c for c in root if c.isalpha())


@lru_cache(maxsize=4)
def lemma_catalog(path, normalize):
    """Candidate evidence for resolving dictionary spellings to corpus lemmas."""
    catalog = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith('('):
            continue
        location, form, tag, features = line.split('\t')
        lemma = re.search(r'(?:^|\|)LEM:([^|]+)', features)
        root = re.search(r'(?:^|\|)ROOT:([^|]+)', features)
        if not lemma or not root:
            continue
        key = root_key(arabic(root[1]))
        word = normalize(arabic(lemma[1]).replace('ٰ', 'ا'))
        item = catalog.setdefault(key, {}).setdefault(word, {'lemma': word, 'part_of_speech': tag, 'examples': []})
        example = arabic(form)
        if example not in item['examples'] and len(item['examples']) < 3:
            item['examples'].append(example)
    return catalog


def lemma_candidates(path, root, normalize):
    # A root narrows candidate selection only. Counts still use one chosen lemma.
    return list(lemma_catalog(str(path), normalize).get(root_key(root), {}).values())

@lru_cache(maxsize=256)
def proper_name_candidates(path, word, normalize):
    """Resolve modern long-vowel spellings only to corpus-tagged proper names."""
    spelling = lambda text: re.sub(r'([وي])\1+', r'\1', normalize(text))
    candidates = set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith('('):
            continue
        location, form, tag, features = line.split('\t')
        if tag != 'PN':
            continue
        lemma = re.search(r'(?:^|\|)LEM:([^|]+)', features)
        if lemma:
            candidate = normalize(arabic(lemma[1]).replace('ٰ', 'ا'))
            if spelling(candidate) == spelling(word):
                candidates.add(candidate)
    return sorted(candidates)

@lru_cache(maxsize=4)
def unprefixed_words(path):
    """Remove only corpus-annotated initial clitics for phrase boundaries."""
    words = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith('('):
            continue
        location, form, tag, features = line.split('\t')
        surah, ayah, position, segment = map(int, location.strip('()').split(':'))
        key = (f'{surah}:{ayah}', position)
        if 'PREFIX' not in features:
            words[key] = words.get(key, '') + arabic(form)
    return words

def install(conn, path, normalize):
    """Import a versioned morphology index independently of vector ingestion."""
    raw = Path(path).read_bytes()
    digest = "lemma-alias-v4:" + hashlib.sha256(raw).hexdigest()
    conn.execute('CREATE TABLE IF NOT EXISTS morphology_version (digest TEXT PRIMARY KEY)')
    conn.execute('CREATE TABLE IF NOT EXISTS morphology (verse_key TEXT, position INTEGER, lemma TEXT, form TEXT, PRIMARY KEY(verse_key, position, lemma))')
    conn.execute('CREATE INDEX IF NOT EXISTS morphology_lemma ON morphology(lemma)')
    conn.execute('CREATE INDEX IF NOT EXISTS morphology_form ON morphology(form)')
    conn.execute('CREATE TABLE IF NOT EXISTS morphology_aliases (form TEXT, lemma TEXT, PRIMARY KEY(form,lemma))')
    if conn.execute('SELECT 1 FROM morphology_version WHERE digest=?', (digest,)).fetchone():
        return
    rows = []
    word_forms = {}
    word_lemmas = {}
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.startswith('('):
            continue
        location, form, tag, features = line.split('\t')
        location_key = tuple(map(int, location.strip('()').split(':')[:3]))
        word_forms.setdefault(location_key, []).append(arabic(form))
        lemma = re.search(r'(?:^|\|)LEM:([^|]+)', features)
        if not lemma and "STEM" not in features:
            continue
        surah, ayah, position, segment = map(int, location.strip('()').split(':'))
        # Expand dagger alif in lemma spellings (e.g. modern عالم instead of علم).
        lemma_text = normalize(arabic(lemma[1] if lemma else form).replace('ٰ', 'ا'))
        rows.append((f'{surah}:{ayah}', position, lemma_text, normalize(arabic(form))))
        word_lemmas.setdefault(location_key, set()).add(lemma_text)
    if not rows:
        raise ValueError('Morphology data has no lemma annotations')
    conn.execute('DELETE FROM morphology')
    conn.executemany('INSERT OR IGNORE INTO morphology VALUES (?, ?, ?, ?)', rows)
    conn.execute('DELETE FROM morphology_aliases')
    aliases = {(normalize(''.join(word_forms[key])), lemma) for key, lemmas in word_lemmas.items() for lemma in lemmas}
    conn.executemany('INSERT INTO morphology_aliases VALUES (?, ?)', aliases)
    conn.execute('DELETE FROM morphology_version')
    conn.execute('INSERT INTO morphology_version VALUES (?)', (digest,))
