"""Rule-based romanization of Indic scripts to business-style Latin.

All 9 Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
Tamil, Telugu, Kannada, Malayalam) -> plain ASCII approximations of how
businesses transliterate their own names (e.g. లక్ష్మీ -> "laxmi-ish",
प्राइवेट -> "praivet"/"praibhet").  Two variants are produced per string when
ambiguous mappings exist (ph/f, sh/s, v/w); downstream matching takes the max
similarity across variant pairs.
"""
import re
import unicodedata

INDIC_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]"
)

DEV = dict(
    vowels="अआइईउऊऋॠऌॡएऐऑओऔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "ri", "ri", "li", "li",
               "e", "ai", "o", "o", "au"],
    cons="कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},   # ph->f, sh->s, v->w (0-based idx)
    matras="ािीुूृेैॉोौ",
    matra_lat=["a", "i", "i", "u", "u", "ri", "e", "ai", "o", "o", "au"],
    virama="्", anusvara="ं", visarga="ः", chandra="ँ",
    digits="०१२३४५६७८९",
    letternames=[("एल", "l"), ("बी", "b"), ("सी", "s"), ("डी", "d"),
                 ("जी", "g"), ("पी", "p"), ("एस", "s"), ("टी", "t"),
                 ("एम", "m"), ("एन", "n"), ("एच", "h"), ("आर", "r"),
                 ("के", "k"), ("एफ", "f"), ("यू", "u"), ("आई", "i"),
                 ("एजी", "g"), ("ओ", "o"), ("ऐ", "a")],
    conjuncts=[("क्ष", "x"), ("श्र", "shr"), ("ज्ञ", "gy")],
)

BEN = dict(
    vowels="অআইঈউঊঋএঐওঔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"],
    cons="কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "j", "r", "l", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "23": "v", "29": "s"},
    matras="ািীুূৃেৈোৌ",
    matra_lat=["a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"],
    virama="্", anusvara="ং", visarga="ঃ", chandra="ঁ",
    digits="০১২৩৪৫৬৭৮৯",
    letternames=[("এল", "l"), ("বি", "b"), ("এস", "s"), ("ডি", "d"),
                 ("জি", "g"), ("পি", "p"), ("টি", "t"), ("এম", "m"),
                 ("এন", "n"), ("আর", "r"), ("কে", "k")],
    conjuncts=[("ক্ষ", "x"), ("শ্র", "shr")],
)

GUR = dict(
    vowels="ਅਆਇਈਉਊਏਐਓਔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    cons="ਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸਹ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "s", "h"],
    cons_alt={"21": "f", "28": "w"},
    matras="ਾਿੀੁੂੇੈੋੌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    virama="੍", anusvara="ਂ", visarga="ਃ", chandra="ੑ",
    digits="੦੧੨੩੪੫੬੭੮੯",
    letternames=[("ਏਲ", "l"), ("ਬੀ", "b"), ("ਸੀ", "s"), ("ਡੀ", "d"),
                 ("ਜੀ", "g"), ("ਪੀ", "p"), ("ਟੀ", "t"), ("ਏਮ", "m"),
                 ("ਏਨ", "n"), ("ਆਰ", "r"), ("ਕੇ", "k")],
    conjuncts=[("ਕ੍ਸ਼", "x"), ("ਸ਼੍ਰ", "shr")],
)

GUJ = dict(
    vowels="અઆઇઈઉઊએઐઓઔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    cons="કખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},
    matras="ાિીુૂેૈોૌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    virama="્", anusvara="ં", visarga="ઃ", chandra="ઁ",
    digits="૦૧૨૩૪૫૬૭૮૯",
    letternames=[("એલ", "l"), ("બી", "b"), ("સી", "s"), ("ડી", "d"),
                 ("જી", "g"), ("પી", "p"), ("ટી", "t"), ("એમ", "m"),
                 ("એન", "n"), ("આર", "r"), ("કે", "k")],
    conjuncts=[("ક્ષ", "x"), ("શ્ર", "shr")],
)

ORY = dict(
    vowels="ଅଆଇଈଉଊଏଐଓଔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    cons="କଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଵଶଷସହ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},
    matras="ାିୀୁୂେୈୋୌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "ai", "o", "au"],
    virama="୍", anusvara="ଂ", visarga="ଃ", chandra="ଁ",
    digits="୦୧୨୩୪୫୬୭୮୯",
    letternames=[("ଏଲ", "l"), ("ବି", "b"), ("ସି", "s"), ("ଡି", "d"),
                 ("ଜି", "g"), ("ପି", "p"), ("ଟି", "t"), ("ଏମ", "m"),
                 ("ଏନ", "n"), ("ଆର", "r"), ("କେ", "k")],
    conjuncts=[("କ୍ଷ", "x"), ("ଶ୍ର", "shr")],
)

TAM = dict(
    vowels="அஆஇஈஉஊஎஏஐஒஓஔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    cons="கஙசஞடணதநபமயரலவழளறனஜஷஸஹ",
    cons_lat=["k", "n", "s", "n", "t", "n", "t", "n", "p", "m",
              "y", "r", "l", "v", "l", "l", "r", "n", "j", "sh", "s", "h"],
    cons_alt={"2": "ch", "19": "s"},
    matras="ாிீுூெேைொோௌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    virama="்", anusvara="ஂ", visarga="ஃ", chandra="",
    digits="௦௧௨௩௪௫௬௭௮௯",
    letternames=[("எல்", "l"), ("பி", "b"), ("எஸ்", "s"), ("டி", "d"),
                 ("ஜி", "g"), ("எம்", "m"), ("என்", "n"), ("ஆர்", "r"),
                 ("கே", "k"), ("சி", "s")],
    conjuncts=[("க்ஷ", "x"), ("ஸ்ரீ", "sri")],
)

TEL = dict(
    vowels="అఆఇఈఉఊఎఏఐఒఓఔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    cons="కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},
    matras="ాిీుూెేైొోౌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    virama="్", anusvara="ం", visarga="ః", chandra="ఁ",
    digits="౦౧౨౩౪౫౬౭౮౯",
    letternames=[("ఎల్", "l"), ("బీ", "b"), ("సీ", "s"), ("డీ", "d"),
                 ("జీ", "g"), ("పీ", "p"), ("టీ", "t"), ("ఎమ్", "m"),
                 ("ఎన్", "n"), ("ఆర్", "r"), ("కే", "k"), ("ఎఫ్", "f")],
    conjuncts=[("క్ష", "x"), ("శ్ర", "shr")],
)

KAN = dict(
    vowels="ಅಆಇಈಉಊಎಏಐಒಓಔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    cons="ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},
    matras="ಾಿೀುೂೆೇೈೊೋೌ",
    matra_lat=["a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    virama="್", anusvara="ಂ", visarga="ಃ", chandra="ಁ",
    digits="೦೧೨೩೪೫೬೭೮೯",
    letternames=[("ಎಲ್", "l"), ("ಬಿ", "b"), ("ಸಿ", "s"), ("ಡಿ", "d"),
                 ("ಜಿ", "g"), ("ಪಿ", "p"), ("ಟಿ", "t"), ("ಎಮ್", "m"),
                 ("ಎನ್", "n"), ("ಆರ್", "r"), ("ಕೆ", "k")],
    conjuncts=[("ಕ್ಷ", "x"), ("ಶ್ರ", "shr")],
)

MAL = dict(
    vowels="അആഇഈഉഊഎഏഐഒഓഔ",
    vowel_lat=["a", "a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    cons="കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹ",
    cons_lat=["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
              "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
              "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"],
    cons_alt={"21": "f", "30": "s", "28": "w"},
    matras="ാിീുൂെേൈൊോൗ",
    matra_lat=["a", "i", "i", "u", "u", "e", "e", "ai", "o", "o", "au"],
    virama="്", anusvara="ം", visarga="ഃ", chandra="",
    digits="൦൧൨൩൪൫൬൭൮൯",
    chillus={"ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k"},
    letternames=[("എൽ", "l"), ("ബി", "b"), ("എസ്", "s"), ("ഡി", "d"),
                 ("ജി", "g"), ("പി", "p"), ("ടി", "t"), ("എം", "m"),
                 ("എൻ", "n"), ("ആർ", "r"), ("കേ", "k")],
    conjuncts=[("ക്ഷ", "x"), ("ശ്ര", "shr")],
)

_SCRIPTS = [DEV, BEN, GUR, GUJ, ORY, TAM, TEL, KAN, MAL]

# nukta-carrier consonants (char -> latin primary, alt)
_NUKTA_CARRIERS = {
    "क": ("q", "k"), "ख": ("kh", "k"), "ग": ("gh", "g"), "ज": ("z", "j"),
    "ड": ("r", "d"), "ढ": ("rh", "dh"), "फ": ("f", "ph"), "य": ("y", "j"),
    "ড": ("r", "d"), "ਜ": ("z", "j"), "ਫ": ("f", "ph"),
}

_CHAR_MAP = {}       # char -> (primary, alt or None)
_NUKTA_SIGNS = set()
for _s in _SCRIPTS:
    for _i, _ch in enumerate(_s["cons"]):
        _lat = _s["cons_lat"][_i] if _i < len(_s["cons_lat"]) else "k"
        _alt = _s["cons_alt"].get(str(_i))
        _CHAR_MAP[_ch] = (_lat, _alt)
    for _i, _ch in enumerate(_s["vowels"]):
        _CHAR_MAP[_ch] = (_s["vowel_lat"][_i], None)
    for _i, _ch in enumerate(_s["matras"]):
        _CHAR_MAP[_ch] = (_s["matra_lat"][_i], None)
    _CHAR_MAP[_s["virama"]] = ("", None)
    if _s["anusvara"]:
        _CHAR_MAP[_s["anusvara"]] = ("n", "m")
    if _s["visarga"]:
        _CHAR_MAP[_s["visarga"]] = ("h", None)
    if _s["chandra"]:
        _CHAR_MAP[_s["chandra"]] = ("n", None)
    for _i, _ch in enumerate(_s["digits"]):
        _CHAR_MAP[_ch] = (str(_i), None)
    for _ch, _lat in _s.get("chillus", {}).items():
        _CHAR_MAP[_ch] = (_lat, None)
    if _s.get("nukta"):
        _NUKTA_SIGNS.add(_s["nukta"])
for _ch in ("\u200c", "\u200d", "\u2060"):
    _CHAR_MAP[_ch] = ("", None)
_CHAR_MAP["।"] = (" ", None)
_CHAR_MAP["॥"] = (" ", None)

# conjunct strings to replace before char mapping
_CONJUNCTS = []
for _s in _SCRIPTS:
    for _pat, _lat in _s["conjuncts"]:
        _CONJUNCTS.append((_pat, _lat))

# helper sets for the lookahead romanizer
_VIRAMA_SET = {_s["virama"] for _s in _SCRIPTS}
_CONS_SET = set()
for _s in _SCRIPTS:
    _CONS_SET.update(_s["cons"])
_MATRA_SET = set()
for _s in _SCRIPTS:
    _MATRA_SET.update(_s["matras"])


def _is_matra_char(ch):
    return ch in _MATRA_SET

# letter names sorted longest-first so greedy replacement works
_LETTERNAMES = []
for _s in _SCRIPTS:
    _LETTERNAMES.extend(_s["letternames"])
_LETTERNAMES.sort(key=lambda x: -len(x[0]))


def _romanize_word(word):
    """Romanize one (possibly mixed) word -> (primary, alt) latin strings.

    Lookahead algorithm: consonant followed by virama -> bare consonant;
    followed by matra -> consonant + matra vowel; otherwise consonant +
    inherent 'a'.  Only trailing *inherent* schwas are deleted, so
    शर्मा -> 'sharma' while लिमिटेड -> 'limited'.
    """
    for pat, lat in _LETTERNAMES:
        while word.startswith(pat):
            word = lat + word[len(pat):]
    for pat, lat in _CONJUNCTS:
        if pat in word:
            word = word.replace(pat, lat)
    prim = []
    altn = []
    prim_inh = False   # last prim vowel was inherent?
    altn_inh = False
    n = len(word)
    i = 0
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        if nxt in _NUKTA_SIGNS and ch in _NUKTA_CARRIERS:
            p, a = _NUKTA_CARRIERS[ch]
            prim.append(p)
            altn.append(a)
            prim_inh = altn_inh = False
            i += 2
            continue
        if ch in _NUKTA_SIGNS:
            i += 1
            continue
        m = _CHAR_MAP.get(ch)
        if m is None:
            if ch.isascii() and ch.isalnum():
                prim.append(ch.lower())
                altn.append(ch.lower())
                prim_inh = altn_inh = False
            i += 1
            continue
        p, a = m
        if ch.isascii() and ch.isalnum():
            # ascii inserted by conjunct pre-pass
            prim.append(p)
            altn.append(a if a is not None else p)
            prim_inh = altn_inh = False
            i += 1
            continue
        _is_cons = ch in _CONS_SET
        if _is_cons:
            nxt_m = _CHAR_MAP.get(nxt)
            nxt_is_virama = nxt in _VIRAMA_SET
            nxt_is_matra = nxt_m is not None and nxt not in _VIRAMA_SET and (
                not nxt.isascii()) and _is_matra_char(nxt)
            if nxt_is_virama:
                prim.append(p)
                altn.append(a if a is not None else p)
                prim_inh = altn_inh = False
                i += 2  # skip consonant + virama
                continue
            if nxt_is_matra:
                vp, va = nxt_m
                va = va if va is not None else vp
                prim.append(p + vp)
                altn.append((a if a is not None else p) + va)
                prim_inh = altn_inh = False
                i += 2  # skip consonant + matra
                continue
            # inherent vowel
            prim.append(p + "a")
            altn.append((a if a is not None else p) + "a")
            prim_inh = altn_inh = True
            i += 1
            continue
        # vowel / mark / digit / other
        prim.append(p)
        altn.append(a if a is not None else p)
        prim_inh = altn_inh = False
        i += 1
    prims = "".join(prim)
    altns = "".join(altn)
    if prim_inh and len(prims) > 2:
        prims = prims[:-1]
    if altn_inh and len(altns) > 2:
        altns = altns[:-1]
    return prims, altns


def romanize(text):
    """Return (primary, alternate) ASCII romanizations of a string."""
    text = unicodedata.normalize("NFKC", text)
    out_p, out_a = [], []
    for word in text.split():
        if not INDIC_RE.search(word):
            out_p.append(word)
            out_a.append(word)
            continue
        p, a = _romanize_word(word)
        out_p.append(p)
        out_a.append(a)
    return " ".join(out_p), " ".join(out_a)


def romanize_variants(text):
    """Return list of distinct romanizations (1 or 2)."""
    p, a = romanize(text)
    if a == p:
        return [p]
    return [p, a]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "लक्ष्मी मार्केटिंग प्राइवेट लिमिटेड",
        "లక్ష్మీ మార్కెటింగ్ ప్రైవేట్ లిమిటెడ్",
        "ಬ್ಲೂ ಸಾಫ್ಟ್\u200cವೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "मॉडर्न फाइनेंस",
        "दत्ता ट्रेडर्स प्राइवेट लिमिटेड",
        "দত্তা ট্রেডার্স প্রাইভেট লিমিটেড",
        "ஸ்ரீ வெங்கடேஷ் ட்ரேடர்ஸ்",
        "കൊച്ചി ട്രേഡഴ്സ്",
        "ਅੰਮ੍ਰਿਤਸਰ ਸਟੀਲ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟੇਡ",
        "ଓଡ଼ିଶା ଟ୍ରେଡର୍ସ",
        "સુરત ટેક્સટાઇલ પ્રાઈવેટ લિમિટેડ",
        "12-11-1178, బౌద్ధ నగర్, సికింద్రాబాద్",
        "दिल्ली",
        "शर्मा ट्रेडर्स",
    ]
    for t in tests:
        p, a = romanize(t)
        print(f"{t!r:55s} -> {p!r} | {a!r}")
