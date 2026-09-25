"""Text normalization + tokenization utilities for entity resolution.

Produces for every record a canonical text view used by both blocking and
feature extraction.  Handles unicode cleanup, accents, legal-suffix folding,
address-word folding, Indic-script romanization (see romanize.py) and
phone-like token canonicalization via double metaphone.
"""
import re
import unicodedata
from doublemetaphone import doublemetaphone

from romanize import romanize, INDIC_RE

# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "private limited", "public limited", "limited liability partnership",
    "limited liability co", "proprietorship", "cooperative society",
    "incorporated", "corporation", "company", "co ltd", "pvt ltd",
    "international", "associates", "association", "enterprises",
    "partnership", "technologies", "technology", "solutions",
    "consultancy", "consulting", "services", "industries", "industrial",
    "traders", "trading", "agencies", "agency", "stores", "imports",
    "exports", "developers", "logistics", "constructions", "contractors",
    "engineering", "enterprises", "holdings", "ventures", "foundation",
    "trust", "society", "committee", "samiti", "sansthan", "mandal",
    "kendra", "seva", "group", "fils", "sas", "sarl", "eurl", "sa",
    "sasu", "sci", "gie", "scm", "snc", "co", "inc", "llc", "llp",
    "ltd", "plc", "pc", "pv", "pr", "corp", "pvt", "priv", "pub",
    "kg", "ag", "gmbh", "ug", "mbh", "oy", "ab", "as", "bv", "nv",
    "spa", "srl", "sarl", "sl", "pte", "sdn", "bhd",
    "limited", "private", "public", "pllc", "inc", "incorporated",
    "company", "companie", "banque", "pharmacy", "pharmacie",
]

LEGAL_SET = frozenset(LEGAL_SUFFIXES)

ADDR_WORD_MAP = {
    "street": "st", "str": "st", "strt": "st", "saint": "st",
    "road": "rd", "rd": "rd",
    "avenue": "av", "ave": "av", "av": "av", "avs": "av",
    "boulevard": "blvd", "boul": "blvd", "blv": "blvd", "blvd": "blvd",
    "drive": "dr", "drv": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "terrace": "ter", "terr": "ter", "ter": "ter",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq",
    "plaza": "plz", "plz": "plz",
    "building": "bldg", "bldg": "bldg", "bld": "bldg", "blg": "bldg",
    "floor": "fl", "flr": "fl", "fl": "fl",
    "apartment": "apt", "apt": "apt",
    "suite": "ste", "ste": "ste",
    "unit": "unit", "u": "unit",
    "room": "rm", "rm": "rm",
    "number": "no", "no": "no", "nbr": "no", "nr": "no", "num": "no",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "n": "n", "s": "s", "e": "e", "w": "w",
    "mount": "mt", "mt": "mt",
    "point": "pt", "pt": "pt",
    "fort": "ft", "ft": "ft",
    "junction": "jct", "jct": "jct",
    "extension": "ext", "ext": "ext",
    "heights": "hts", "hts": "hts",
    "crossing": "xing", "xing": "xing",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "market": "mkt", "mkt": "mkt",
    "sector": "sec", "sec": "sec",
    "phase": "ph", "ph": "ph",
    "plot": "plot", "plotno": "plot", "sy": "plot", "survey": "plot",
    "khasra": "khasra", "kh": "khasra", "khno": "khasra",
    "house": "house", "hs": "house",
    "block": "block", "blk": "block",
    "society": "soc", "soc": "soc", "society": "soc",
    "nagar": "nagar", "colony": "colony", "col": "colony",
    "village": "village", "vill": "village", "vlg": "village", "village": "village",
    "post": "po", "po": "po", "office": "po",
    "pin": "pin", "pincode": "pin", "zip": "pin",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp",
    "behind": "bhnd", "beside": "bside", "adjacent": "adj",
    "and": "&", "&": "&",
    "rue": "rue", "r": "rue", "avenuedelaru": "rue",
    "allée": "allee", "allee": "allee",
    "boulevarddupres": "blvd",
    "chemin": "chem", "chem": "chem",
    "place": "pl",
    "quai": "quai", "impasse": "imp", "imp": "imp",
    "résidence": "res", "residence": "res", "res": "res",
    "résid": "res",
    "imm": "bldg", "immeuble": "bldg",
    "lot": "lot", "lotissement": "lot",
    "étage": "fl", "etage": "fl",
    "appt": "apt",
}

STATE_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "or", "orissa": "or", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "puducherry": "py", "pondicherry": "py",
    "jammu kashmir": "jk", "jammu and kashmir": "jk",
    "ladakh": "la", "ladakh union territory": "la",
    "andaman nicobar": "an", "andaman and nicobar": "an",
    "dadra nagar haveli": "dn", "daman diu": "dd",
    "daman and diu": "dd", "lakshadweep": "ld",
    # France
    "ile de france": "idf", "auvergne rhone alpes": "ara",
    "bourgogne franche comte": "bfc", "brittany": "bre", "bretagne": "bre",
    "centre val de loire": "cvl", "corsica": "cor", "corse": "cor",
    "grand est": "ges", "hauts de france": "hdf", "normandy": "nor",
    "normandie": "nor", "nouvelle aquitaine": "naq",
    "occitanie": "occ", "pays de la loire": "pdl",
    "provence alpes cote d azur": "pac", "provence alpes cote dazur": "pac",
}

# devanagari state names in their own script (address side)
INDIC_STATE_WORDS = {
    "दिल्ली": "dl", "दिल्ली नई": "dl",
    "महाराष्ट्र": "mh", "कर्नाटक": "ka", "तमिलनाडु": "tn",
    "तेलंगाना": "tg", "आंध्र प्रदेश": "ap", "गुजरात": "gj",
    "राजस्थान": "rj", "उत्तर प्रदेश": "up", "मध्य प्रदेश": "mp",
    "पश्चिम बंगाल": "wb", "हरियाणा": "hr", "पंजाब": "pb",
    "केरल": "kl", "बिहार": "br", "ओडिशा": "or", "उत्तराखंड": "uk",
    "छत्तीसगढ": "cg", "झारखंड": "jh", "गोवा": "ga",
}

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9&]+")

STATE_ABBR_INV = frozenset(STATE_ABBR.values())


def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def basic_clean(s):
    """NFKC, accent-strip, lowercase, collapse punctuation to spaces."""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s)
    s = s.lower()
    s = s.replace("&", " & ")
    s = _NONWORD_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


PHONE_CACHE = {}


def phone_token(w):
    """Double-metaphone primary of a token (cached)."""
    v = PHONE_CACHE.get(w)
    if v is None:
        v = doublemetaphone(w)[0]
        PHONE_CACHE[w] = v
    return v


class Record:
    """Canonical views of one business record."""
    __slots__ = (
        "eid", "src", "country", "name_raw", "addr_raw",
        "name", "addr", "name_r", "addr_r", "name_rt", "addr_rt",
        "name_t", "addr_t",
        "name_toks", "addr_toks", "name_rt_toks", "addr_rt_toks",
        "name_ph", "addr_ph", "name_core", "name_core_toks",
        "addr_digits", "addr_pin", "addr_state", "domain", "sig",
    )

    def __init__(self, eid, src, country, name_raw, addr_raw):
        self.eid = eid
        self.src = src
        self.country = country
        self.name_raw = name_raw
        self.addr_raw = addr_raw
        self._build()

    def _build(self):
        name = self.name_raw
        addr = self.addr_raw
        # romanize indic scripts once per record
        if INDIC_RE.search(name) or INDIC_RE.search(addr):
            np_, na = romanize(name)
            ap_, aa = romanize(addr)
            self.name_r, self.addr_r = np_, ap_
            self.name_rt, self.addr_rt = na, aa
        else:
            self.name_r = self.name_rt = None
            self.addr_r = self.addr_rt = None
        # latin view: raw if latin, romanized otherwise (primary variant)
        lat_name = self.name_r if self.name_r is not None else name
        lat_addr = self.addr_r if self.addr_r is not None else addr
        self.name = basic_clean(lat_name)
        self.addr = basic_clean(lat_addr)
        # alternate romanization view (suffix _t)
        if self.name_rt is not None:
            self.name_t = basic_clean(self.name_rt)
            self.addr_t = basic_clean(self.addr_rt)
        else:
            self.name_t = self.name
            self.addr_t = self.addr
        self.name_toks = self.name.split()
        self.addr_toks = self.addr.split()
        self.name_rt_toks = self.name_t.split()
        self.addr_rt_toks = self.addr_t.split()
        # phone-like tokens (primary romanization only)
        self.name_ph = tuple(phone_token(t) for t in self.name_toks)
        self.addr_ph = tuple(phone_token(t) for t in self.addr_toks)
        # core name: drop legal suffix tokens for a stronger key
        self.name_core_toks = [t for t in self.name_toks if t not in LEGAL_SET]
        self.name_core = " ".join(self.name_core_toks)
        # digits in address (house numbers etc.)
        self.addr_digits = re.findall(r"\d+", self.addr_raw)
        # pin code: last 5-6 digit token in the address
        pin = ""
        for tok in re.findall(r"\d{5,6}", addr):
            pin = tok
        self.addr_pin = pin
        # state abbreviation if present
        self.addr_state = ""
        addr_l = strip_accents(addr).lower()
        for full, ab in STATE_ABBR.items():
            if full in addr_l:
                self.addr_state = ab
                break
        if not self.addr_state:
            for w, ab in INDIC_STATE_WORDS.items():
                if w in addr:
                    self.addr_state = ab
                    break
        if not self.addr_state:
            # abbreviation token in the last 3 tokens of the address
            tail = self.addr_toks[-3:]
            for t in reversed(tail):
                if t in STATE_ABBR_INV:
                    self.addr_state = t
                    break
        # domain
        m = re.search(r"([a-z0-9\-]+)\.(com|net|org|co|in|fr)\b", name.lower())
        self.domain = m.group(1) if m else ""
        # structural signature for blocking: digits of name + addr tokens set
        self.sig = (tuple(sorted(set(self.name_toks))),
                    tuple(sorted(set(self.addr_toks))))


def build_record(eid, src, country, name, addr):
    return Record(eid, src, country, name, addr)
