"""Thumbnail generator for the Price & Power channel.

Written to the channel owner's brief, which replaced an earlier version of
this file that was rejected - correctly. That version drew dark navy cards
with a metric eyebrow, two state names, two values, a gap pill and a verdict
bar: six elements, a chart rather than a thumbnail, and its hook line
("THE CHEAPER HOME LOSES") answered the question on the image, so there was
nothing left to click for.

The brief, applied here:

  ONE CONFLICT PER FRAME.        Two subjects, two numbers, one hook. Nothing else.
  SHOW THE PROBLEM, HIDE THE     The hook poses a question. It never names the
  ANSWER.                        winner and never restates the title.
  PHOTOREAL, CINEMATIC, SIMPLE.  Real photographs, graded warm against cool so
                                 the two sides separate instantly.
  HIERARCHY.                     number -> which side is which -> hook.
  NO TITLE DUPLICATION.          The title carries the detail; the image takes
                                 the attention. So no metric label on the frame.

Engineering constraints, unchanged:
  - Railway has no Chromium, so this is SVG rasterised by cairosvg, the same
    path visuals.py uses in production.
  - Text is MEASURED against the real font file through PIL and shrunk to fit;
    a character-count estimate puts 170px type off the edge of the frame.
  - Nothing below MIN_TEXT. At a 120px-wide preview - the true size in a phone
    feed - smaller elements dissolve into mush.
"""

import base64
import hashlib
import os
import re

try:
    from PIL import ImageFont
except Exception:                                    # pragma: no cover
    ImageFont = None

try:
    import requests
except Exception:                                    # pragma: no cover
    requests = None

import cairosvg

W, H = 1280, 720

INK = "#ffffff"
SHADE = "#080d16"          # the dark used for scrims, never as a flat ground
WARM = "#ff9d2e"           # left-side grade
COOL = "#3f9bff"           # right-side grade

MIN_TEXT = 56              # unreadable below this in a phone feed
BADGE = (1112, 656, 1272, 712)      # YouTube stamps the duration here

PHOTO_DIR = os.environ.get("PHOTO_CACHE_DIR", "/tmp/thumb_photos")
try:
    os.makedirs(PHOTO_DIR, exist_ok=True)
except Exception:
    pass

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Black.ttf",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_FP = _font_path()
# The family named in the SVG must resolve to the face PIL measured, or the
# fit is a lie: cairosvg would lay out with a different face and text that
# fitted locally would run off the frame on the server.
if _FP and "Roboto-Black" in _FP:
    FAMILY = "Roboto Black, Roboto, DejaVu Sans, sans-serif"
elif _FP and "RobotoCondensed" in _FP:
    FAMILY = "Roboto Condensed, Roboto, DejaVu Sans, sans-serif"
else:
    FAMILY = "DejaVu Sans, sans-serif"

_FCACHE = {}


def _load(size):
    if ImageFont is None or not _FP:
        return None
    k = int(size)
    if k not in _FCACHE:
        try:
            _FCACHE[k] = ImageFont.truetype(_FP, k)
        except Exception:
            _FCACHE[k] = None
    return _FCACHE[k]


def text_width(s, size):
    f = _load(size)
    if f is None:
        return len(str(s)) * size * 0.62          # errs wide, so text shrinks
    try:
        b = f.getbbox(str(s))
        return b[2] - b[0]
    except Exception:
        return len(str(s)) * size * 0.62


def fit(s, max_w, start, min_size=MIN_TEXT, step=2):
    size = int(start)
    while size > min_size and text_width(s, size) > max_w:
        size -= step
    return max(int(min_size), size)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ------------------------------------------------------------------ numbers
def money(n):
    """Compact enough to be huge on screen: 465000 -> $465K."""
    n = float(n)
    neg, n = n < 0, abs(float(n))
    if n >= 1_000_000:
        s = ("$%.1fM" % (n / 1_000_000)).replace(".0M", "M")
    elif n >= 10_000:
        s = "$%dK" % round(n / 1000)
    elif n >= 1000:
        s = "$%s" % format(int(round(n)), ",")
    else:
        s = "$%d" % round(n)
    return ("-" if neg else "") + s


def short_name(name, limit=13):
    n = str(name).strip()
    if len(n) <= limit:
        return n.upper()
    parts = n.split()
    if len(parts) >= 2:
        c = parts[0][0] + ". " + " ".join(parts[1:])
        if len(c) <= limit:
            return c.upper()
        c = "".join(p[0] for p in parts)              # United Kingdom -> UK
        if len(c) >= 2:
            return c.upper()
    return n[:limit].upper()


# ------------------------------------------------------------------ photos
# Same fetch-and-cache shape visuals.py already runs in production: one call
# per query, cached to disk, and a clean None on any failure so a missing
# photo degrades the design instead of breaking the run.
_PHOTO_MEM = {}

# Order matters and so do word boundaries. The first version listed the
# housing pattern first and matched on a bare "house", so "median HOUSEhold
# income" was drawn as a suburban street - the wrong story entirely. Income
# is tested before housing now, and every term is boundaried.
SUBJECT_QUERIES = [
    (r"\b(income|salary|salaries|wage|wages|pay|earnings|paycheck)\b", "%s downtown skyline morning"),
    (r"\b(childcare|daycare|nursery|preschool)\b",   "empty nursery classroom"),
    (r"\b(childbirth|maternity|birth)\b",            "hospital maternity room"),
    (r"\b(health|insurance|medical|hospital|premium)\b", "hospital corridor empty"),
    (r"\b(grocer|groceries|food|supermarket)\b",     "grocery store aisle full shopping cart"),
    (r"\b(electric|electricity|power|utility|utilities|energy)\b", "home electricity meter close up"),
    (r"\b(rent|apartment|lease|renting)\b",          "%s apartment building exterior"),
    (r"\b(home|homes|house|houses|housing|property|mortgage)\b", "%s suburban house exterior"),
    (r"\btax(es)?\b",                                "%s suburban street houses"),
    (r"\b(commute|traffic|transport|fuel|gasoline|petrol)\b", "%s highway traffic"),
]


def photo_query(entity, subject):
    """A query the search will actually answer well.

    A bare place name returns postcards - mountains, sunsets, license plates -
    the same travel-brochure look the scripts were cleaned of. The subject has
    to be in the query or the picture tells the wrong story.
    """
    s = str(subject or "").lower()
    for pat, tpl in SUBJECT_QUERIES:
        if re.search(pat, s):
            return tpl % entity if "%s" in tpl else tpl
    return "%s residential neighborhood" % entity


def is_place_specific(subject):
    """True when the query differs per side, so a split shows two pictures.

    Some subjects have no per-place image worth searching - a trolley of
    groceries looks the same in Idaho and in Kent. Those returned the SAME
    photo for both halves, which drew a split screen of one identical picture
    twice. Those subjects get the single-photograph layout instead.
    """
    s = str(subject or "").lower()
    for pat, tpl in SUBJECT_QUERIES:
        if re.search(pat, s):
            return "%s" in tpl
    return True


def fetch_photo(query):
    """JPEG bytes for a query, or None. Never raises."""
    if not query:
        return None
    if query in _PHOTO_MEM:
        return _PHOTO_MEM[query]
    disk = os.path.join(PHOTO_DIR, hashlib.md5(query.encode()).hexdigest() + ".jpg")
    raw = None
    if os.path.exists(disk):
        try:
            with open(disk, "rb") as f:
                raw = f.read()
        except Exception:
            raw = None
    if raw is None:
        key = os.environ.get("PEXELS_API_KEY")
        if not key or requests is None:
            return None
        try:
            r = requests.get("https://api.pexels.com/v1/search",
                             params={"query": query, "per_page": 1,
                                     "orientation": "landscape", "size": "large"},
                             headers={"Authorization": key}, timeout=25)
            r.raise_for_status()
            photos = r.json().get("photos") or []
            if not photos:
                return None
            src = photos[0]["src"].get("large2x") or photos[0]["src"].get("large")
            im = requests.get(src, timeout=40)
            im.raise_for_status()
            raw = im.content
            try:
                with open(disk, "wb") as f:
                    f.write(raw)
            except Exception:
                pass
        except Exception:
            return None
    if not raw or len(raw) < 2000:
        return None
    _PHOTO_MEM[query] = raw
    return raw


def _uri(raw):
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


# ------------------------------------------------------------------ drawing
_EMITTED = []


def _t(x, y, s, size, fill=INK, anchor="middle"):
    tw = text_width(s, size)
    if anchor == "middle":
        x0, x1 = x - tw / 2.0, x + tw / 2.0
    elif anchor == "end":
        x0, x1 = x - tw, x
    else:
        x0, x1 = x, x + tw
    _EMITTED.append({"text": str(s), "size": size, "x0": x0, "x1": x1,
                     "y0": y - size * 0.76, "y1": y + size * 0.24})
    # Heavy type on a photograph needs its own edge or it vanishes the moment
    # the picture behind it goes pale. Drawn twice: a dark stroke, then fill.
    common = ('font-family="%s" font-size="%d" font-weight="900" text-anchor="%s"'
              % (FAMILY, size, anchor))
    return ('<text x="%d" y="%d" %s fill="none" stroke="%s" stroke-width="%d" '
            'stroke-linejoin="round" opacity="0.85">%s</text>'
            '<text x="%d" y="%d" %s fill="%s">%s</text>'
            % (x, y, common, SHADE, max(5, int(size * 0.075)), esc(s),
               x, y, common, fill, esc(s)))


def validate(margin=26):
    out = []
    for e in _EMITTED:
        if e["x0"] < margin or e["x1"] > W - margin:
            out.append('"%s" (%dpx) runs off the frame: x %.0f..%.0f'
                       % (e["text"][:26], e["size"], e["x0"], e["x1"]))
        if e["y0"] < 0 or e["y1"] > H:
            out.append('"%s" runs off vertically: y %.0f..%.0f'
                       % (e["text"][:26], e["y0"], e["y1"]))
        if e["size"] < MIN_TEXT:
            out.append('"%s" is %dpx, under the %dpx floor' % (e["text"][:26], e["size"], MIN_TEXT))
    return out


def _defs():
    return (
        '<defs>'
        # bottom scrim: the hook always sits on something dark
        '<linearGradient id="hook" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="%s" stop-opacity="0"/>'
        '<stop offset="0.45" stop-color="%s" stop-opacity="0.80"/>'
        '<stop offset="1" stop-color="%s" stop-opacity="0.96"/></linearGradient>'
        # top scrim, lighter: the numbers need contrast without hiding the photo
        '<linearGradient id="top" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="%s" stop-opacity="0.62"/>'
        '<stop offset="1" stop-color="%s" stop-opacity="0.10"/></linearGradient>'
        '</defs>' % (SHADE, SHADE, SHADE, SHADE, SHADE)
    )


def _photo_half(raw, x, w, tint, tint_op=0.20):
    """A photograph filling one half, graded toward warm or cool."""
    if raw:
        img = ('<image x="%d" y="0" width="%d" height="%d" xlink:href="%s" '
               'preserveAspectRatio="xMidYMid slice"/>' % (x, w, H, _uri(raw)))
    else:
        # no photo: a deep graded panel, still cinematic, never a flat block
        img = ('<rect x="%d" y="0" width="%d" height="%d" fill="%s"/>'
               '<rect x="%d" y="0" width="%d" height="%d" fill="%s" opacity="0.30"/>'
               % (x, w, H, SHADE, x, w, H, tint))
    return (img + '<rect x="%d" y="0" width="%d" height="%d" fill="%s" opacity="%.2f"/>'
            % (x, w, H, tint, tint_op))


def _hook_band(hook, top=498):
    size = fit(hook, W - 110, 108, MIN_TEXT)
    return ('<rect x="0" y="%d" width="%d" height="%d" fill="url(#hook)"/>' % (top, W, H - top)
            + _t(W // 2, 646, hook, size))


# ------------------------------------------------------------------ layouts
def _lay_split(d):
    """Two photographs, hard vertical divide. The default and the clearest."""
    half = W // 2
    la, lb = short_name(d["a"]), short_name(d["b"])
    ns = min(fit(la, half - 90, 74, MIN_TEXT), fit(lb, half - 90, 74, MIN_TEXT))
    va, vb = money(d["va"]), money(d["vb"])
    vs = min(fit(va, half - 70, 184, 96), fit(vb, half - 70, 184, 96))
    return "".join([
        _defs(),
        _photo_half(d["pa"], 0, half, WARM),
        _photo_half(d["pb"], half, W - half, COOL),
        '<rect x="0" y="0" width="%d" height="250" fill="url(#top)"/>' % W,
        '<rect x="%d" y="0" width="8" height="%d" fill="%s" opacity="0.92"/>' % (half - 4, H, SHADE),
        _t(half // 2, 150, la, ns),
        _t(half // 2, 316, va, vs),
        _t(half + half // 2, 150, lb, ns),
        _t(half + half // 2, 316, vb, vs),
        _hook_band(d["hook"]),
    ])


def _lay_diagonal(d):
    """Same story, different construction - the brief asks the shape to vary."""
    half = W // 2
    la, lb = short_name(d["a"]), short_name(d["b"])
    ns = min(fit(la, 470, 70, MIN_TEXT), fit(lb, 470, 70, MIN_TEXT))
    va, vb = money(d["va"]), money(d["vb"])
    vs = min(fit(va, 470, 172, 96), fit(vb, 470, 172, 96))
    return "".join([
        _defs(),
        '<clipPath id="cl"><polygon points="0,0 720,0 560,720 0,720"/></clipPath>',
        '<clipPath id="cr"><polygon points="720,0 1280,0 1280,720 560,720"/></clipPath>',
        '<g clip-path="url(#cl)">%s</g>' % _photo_half(d["pa"], 0, 760, WARM),
        '<g clip-path="url(#cr)">%s</g>' % _photo_half(d["pb"], 520, W - 520, COOL),
        '<rect x="0" y="0" width="%d" height="250" fill="url(#top)"/>' % W,
        '<polygon points="724,0 764,0 604,720 564,720" fill="%s" opacity="0.92"/>' % SHADE,
        _t(292, 148, la, ns),
        _t(292, 306, va, vs),
        _t(958, 148, lb, ns),
        _t(958, 306, vb, vs),
        _hook_band(d["hook"]),
    ])


def _lay_hero(d):
    """One photograph, both numbers over it. For when a single image carries
    the whole idea - a full trolley, a meter, a maternity room."""
    raw = d["pa"] or d["pb"]
    la, lb = short_name(d["a"]), short_name(d["b"])
    ns = min(fit(la, 480, 66, MIN_TEXT), fit(lb, 480, 66, MIN_TEXT))
    va, vb = money(d["va"]), money(d["vb"])
    vs = min(fit(va, 470, 176, 96), fit(vb, 470, 176, 96))
    return "".join([
        _defs(),
        _photo_half(raw, 0, W, WARM, tint_op=0.12),
        '<rect x="0" y="0" width="%d" height="%d" fill="%s" opacity="0.42"/>' % (W, H, SHADE),
        '<rect x="0" y="0" width="%d" height="300" fill="url(#top)"/>' % W,
        _t(310, 150, la, ns),
        _t(310, 322, va, vs),
        _t(970, 150, lb, ns),
        _t(970, 322, vb, vs),
        '<rect x="%d" y="188" width="7" height="150" fill="%s" opacity="0.9"/>' % (W // 2 - 3, INK),
        _hook_band(d["hook"]),
    ])


LAYOUTS = {"split": _lay_split, "diagonal": _lay_diagonal, "hero": _lay_hero}


def choose_layout(d):
    """Vary the construction, but never at the cost of clarity.

    Two photographs support a split or a diagonal; one photograph only
    supports the hero. The hash rotates between whatever is genuinely
    available so sixty videos do not share one composition.
    """
    # Two DIFFERENT photographs are what a split needs. Identical ones - which
    # is what a subject-only query returns for both sides - make a split screen
    # of the same picture twice.
    two_pictures = bool(d["pa"]) and bool(d["pb"]) and d["qa"] != d["qb"]
    if two_pictures:
        eligible = ["split", "diagonal"]
    elif d["pa"] or d["pb"]:
        eligible = ["hero"]                  # one picture, so draw it once
    else:
        # No photograph at all - Pexels down, no key, nothing cached. The hero
        # fallback painted one flat tinted panel and looked like mud. A split
        # of the two graded panels still separates warm from cool, which is
        # the one thing carrying the comparison when there is no picture.
        eligible = ["split"]
    seed = hashlib.md5(("%s|%s|%s" % (d["a"], d["b"], d["subject"])).encode()).hexdigest()
    return eligible[int(seed[:8], 16) % len(eligible)]


# ------------------------------------------------------------------ spec
_ANSWER_WORDS = re.compile(
    r"\b(wins?|loses?|beats?|cheaper|pricier|costlier|winner|better|worse|"
    r"ahead|behind|richer|poorer)\b", re.I)


# A repeated phrase stops working within about three uploads, so the fallback
# rotates instead of stamping one line on every video that needs it.
_FALLBACKS = ["WHO ACTUALLY WINS?", "NOT WHAT YOU THINK", "WORTH IT?",
              "CHEAPER = EASIER?", "WHO CAN AFFORD IT?", "THE REAL COST"]


def clean_hook(hook, a, b):
    """The hook must pose the question, not settle it.

    The first version of this rejected any hook containing "wins", "cheaper"
    or "loses" - and so threw out "WHO ACTUALLY WINS?" and "CHEAPER =
    EASIER?", both of which are exactly right. The word is not the problem;
    the GRAMMAR is. A question opens a gap. A flat statement of the result
    closes it, which is what "THE CHEAPER HOME LOSES" did.

    So: an answer-word is only a defect in a sentence that ASSERTS - one with
    no question mark. Naming either place is always a defect; that is the
    title's job, and repeating it on the image wastes the frame.
    """
    h = re.sub(r"\s+", " ", str(hook or "")).strip().upper().strip(".!")
    words = [w for w in re.split(r"\s+", h) if w]
    asks = h.endswith("?")
    bad = (not h
           or len(words) > 4
           or (not asks and _ANSWER_WORDS.search(h))
           or str(a).upper() in h
           or str(b).upper() in h)
    if not bad:
        return h
    seed = int(hashlib.md5(("%s|%s" % (a, b)).encode()).hexdigest()[:8], 16)
    return _FALLBACKS[seed % len(_FALLBACKS)]


def build_spec(payload):
    a = payload.get("entity_a") or "A"
    b = payload.get("entity_b") or "B"
    va = float(payload.get("value_a") or 0)
    vb = float(payload.get("value_b") or 0)
    subject = payload.get("subject") or "cost of living"

    qa = payload.get("photo_query_a") or photo_query(a, subject)
    qb = payload.get("photo_query_b") or photo_query(b, subject)
    if not is_place_specific(subject):
        qb = qa                      # one picture, drawn once, hero layout

    return {
        "a": a, "b": b, "va": va, "vb": vb, "subject": subject,
        "hook": clean_hook(payload.get("hook"), a, b),
        "pa": fetch_photo(qa), "pb": fetch_photo(qb),
        "qa": qa, "qb": qb,
        "layout": payload.get("layout"),
    }


def render_svg(payload):
    del _EMITTED[:]
    d = build_spec(payload)
    name = d.get("layout") or choose_layout(d)
    if name not in LAYOUTS:
        name = "split"
    body = LAYOUTS[name](d)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
           'xmlns:xlink="http://www.w3.org/1999/xlink" '
           'width="%d" height="%d" viewBox="0 0 %d %d">%s</svg>'
           % (W, H, W, H, body))
    return svg, name, d


def render_png(payload, out_path=None):
    svg, name, d = render_svg(payload)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                           output_width=W, output_height=H)
    if out_path:
        with open(out_path, "wb") as f:
            f.write(png)
    return png, name, d
