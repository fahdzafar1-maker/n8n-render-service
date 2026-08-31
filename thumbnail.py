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

# Semantic colour for the two figures. This is safe against the "hide the
# answer" rule: it marks which PRICE is lower, which anyone can already see by
# reading the digits. It does not say which side leaves you richer overall -
# that verdict is the video, and the frame still withholds it.
HIGH = "#ff6b5e"           # the worse figure for the viewer
LOW  = "#4ade80"           # the better figure for the viewer

# For a COST, the bigger number is the bad one. For INCOME it is the good one -
# and the first version coloured Utah's $97K red against Ohio's $72K green on a
# median-household-income comparison, which reads as exactly backwards.
EARNINGS_RE = re.compile(
    r"\b(income|salary|salaries|wage|wages|pay|earnings|paycheck|takehome|"
    r"take-home|savings|surplus)\b", re.I)


def bigger_is_better(subject):
    return bool(EARNINGS_RE.search(str(subject or "")))

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



# ------------------------------------------------------------------ icons
# "No visual hook - just text and numbers" was the verdict on the first cut,
# and it was right: a frame of type reads as a slide, not a video. These are
# small, flat, high-contrast glyphs that survive a 120px preview, drawn under
# each number so the eye lands on a SHAPE before it reads a digit.
def _reg_icon(cx, cy, s):
    _BOXES.append({"text": "[icon]", "size": MIN_TEXT,
                   "x0": cx - s, "x1": cx + s, "y0": cy - s * 0.62, "y1": cy + s * 0.62})


def _icon_house(cx, cy, s, fill):
    _reg_icon(cx, cy, s)
    w = s * 0.92
    return ('<path d="M%.0f %.0f L%.0f %.0f L%.0f %.0f L%.0f %.0f L%.0f %.0f '
            'L%.0f %.0f L%.0f %.0f Z" fill="%s" stroke="%s" stroke-width="6" '
            'stroke-linejoin="round"/>'
            % (cx, cy - s * 0.52, cx + w, cy + s * 0.02, cx + w * 0.72, cy + s * 0.02,
               cx + w * 0.72, cy + s * 0.52, cx - w * 0.72, cy + s * 0.52,
               cx - w * 0.72, cy + s * 0.02, cx - w, cy + s * 0.02, fill, SHADE))


def _icon_coins(cx, cy, s, fill):
    _reg_icon(cx, cy, s)
    out = []
    for i, dy in enumerate((s * 0.34, 0.0, -s * 0.34)):
        out.append('<ellipse cx="%.0f" cy="%.0f" rx="%.0f" ry="%.0f" fill="%s" '
                   'stroke="%s" stroke-width="5"/>'
                   % (cx, cy + dy, s * 0.78, s * 0.24, fill, SHADE))
    return "".join(out)


def _icon_bill(cx, cy, s, fill):
    _reg_icon(cx, cy, s)
    return ('<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" rx="6" fill="%s" '
            'stroke="%s" stroke-width="6"/>'
            '<circle cx="%.0f" cy="%.0f" r="%.0f" fill="%s"/>'
            % (cx - s * 0.92, cy - s * 0.46, s * 1.84, s * 0.92, fill, SHADE,
               cx, cy, s * 0.24, SHADE))


ICONS = {"house": _icon_house, "coins": _icon_coins, "bill": _icon_bill}

ICON_FOR = [
    (r"\b(income|salary|wage|wages|pay|earnings)\b", "coins"),
    (r"\b(home|house|housing|property|mortgage|rent|apartment)\b", "house"),
    (r"\b(tax|taxes|insurance|premium|bill|electric|utility|grocer)\b", "bill"),
]


def icon_for(subject):
    s = str(subject or "").lower()
    for pat, name in ICON_FOR:
        if re.search(pat, s):
            return name
    return "bill"


def _vs_badge(cx, cy, r=72):
    """The middle of a split screen is dead space; a VS badge gives the eye a
    pivot and says 'comparison' before a single word is read."""
    return ('<circle cx="%d" cy="%d" r="%d" fill="%s" stroke="%s" stroke-width="8"/>'
            % (cx, cy, r, SHADE, WARM)
            + _t(cx, cy + 24, "VS", 66, WARM))


def _context_label(text):
    """A short category word - HOME PRICES, POWER BILL.

    Deliberately re-added after being cut. The brief says do not duplicate the
    title, and that stands: this is not the title, it is the missing CONTEXT.
    Without it the review was blunt - "are these home prices? salaries?
    taxes? - without context the numbers feel random."
    """
    t = str(text or "").upper()
    t = re.sub(r"\b(MEDIAN|AVERAGE|MONTHLY|ANNUAL|PER MONTH|PER YEAR)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip() or "COST OF LIVING"
    # "Property tax on a median home" truncated to "PROPERTY TAX ON" - a label
    # ending on a preposition reads as a sentence that got cut off.
    t = re.sub(r"\b(ON|OF|FOR|PER|A|AN|THE|IN)\b\s*$", "", t).strip()
    words = t.split()
    if len(words) > 3:
        t = " ".join(words[:3])
    return re.sub(r"\b(ON|OF|FOR|PER|A|AN|THE|IN)\b\s*$", "", t).strip()


# ------------------------------------------------------------------ drawing
_EMITTED = []
_BOXES = []          # non-text elements that must not sit under type


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


def _overlap(a, b):
    return not (a["x1"] <= b["x0"] or b["x1"] <= a["x0"]
                or a["y1"] <= b["y0"] or b["y1"] <= a["y0"])


def validate(margin=26):
    """Off-frame, under-size, AND colliding.

    The overlap test was added after a render put "$465K" straight through the
    VS badge and the state name through the top of the figure - both passed
    every other check, because each element on its own was fine.
    """
    out = []
    everything = _EMITTED + _BOXES
    for i, a in enumerate(everything):
        for b in everything[i + 1:]:
            if _overlap(a, b):
                out.append('"%s" overlaps "%s"' % (a["text"][:18], b["text"][:18]))
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
        # This used to end at 0.10 opacity and then stop dead, which drew a
        # visible horizontal seam right across the photograph at y=299 - a
        # 22-level step, measured on the first live render. It fades to zero
        # now, and the darkening it provides for the numbers is carried by
        # the middle stop instead of by the cut-off.
        '<linearGradient id="top" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="%s" stop-opacity="0.70"/>'
        '<stop offset="0.62" stop-color="%s" stop-opacity="0.30"/>'
        '<stop offset="1" stop-color="%s" stop-opacity="0"/></linearGradient>'
        '</defs>' % (SHADE, SHADE, SHADE, SHADE, SHADE, SHADE)
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


_EMPH = re.compile(r"\b(REALLY|ACTUALLY|TRULY|EVER|WORTH|NOT)\b")


_FACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face.png")
_FACE_CACHE = {}


def face_layer(x_right=True, height=330):
    """The host's cut-out, bottom-anchored at one side.

    Kept deliberately small. A face raises click-through, but the numbers are
    still the thing being sold, and a half-frame portrait buries them. Absent
    file = absent layer, silently: no face must never mean no thumbnail.
    """
    if not os.path.exists(_FACE_PATH):
        return "", 0
    if "uri" not in _FACE_CACHE:
        try:
            with open(_FACE_PATH, "rb") as f:
                raw = f.read()
            _FACE_CACHE["uri"] = "data:image/png;base64," + base64.b64encode(raw).decode()
            from PIL import Image as _I
            _FACE_CACHE["size"] = _I.open(_FACE_PATH).size
        except Exception:
            _FACE_CACHE["uri"] = None
    if not _FACE_CACHE.get("uri"):
        return "", 0
    fw, fh = _FACE_CACHE["size"]
    w = int(height * fw / float(fh))
    x = (W - w - 8) if x_right else 8
    y = H - height
    # Registered as a box so validate() can catch it colliding with type.
    # The first cut sat the face across the bottom of "$339K": every text
    # check passed, because the collision was text against IMAGE and nothing
    # was looking for that.
    _BOXES.append({"text": "[face]", "size": MIN_TEXT, "x0": x, "x1": x + w,
                   "y0": y, "y1": y + height})
    return ('<image x="%d" y="%d" width="%d" height="%d" xlink:href="%s" '
            'preserveAspectRatio="xMidYMax meet"/>' % (x, y, w, height, _FACE_CACHE["uri"]),
            w)


def _hook_band(hook, top=498, avoid_right=0):
    """The hook, with one word carried in amber.

    Flat white across four words gives the eye nowhere to land. Emphasis goes
    on the sceptical word - REALLY, ACTUALLY, NOT - because that is where the
    doubt lives; failing that, on the last word before the question mark.
    """
    words = [w for w in str(hook).split(" ") if w]
    avail = W - 110 - avoid_right
    size = fit(hook, avail, 108, MIN_TEXT)
    hi = -1
    for i, w in enumerate(words):
        if _EMPH.search(w):
            hi = i
            break
    if hi < 0 and len(words) > 1:
        hi = len(words) - 1

    # PIL reports a space as near-zero for many faces, so words ran together
    # ("REALLY BETTER?" read as one word). Floor it.
    space = max(text_width(" ", size), size * 0.30)
    widths = [text_width(w, size) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    x = (W - avoid_right - total) / 2.0
    out = ['<rect x="0" y="%d" width="%d" height="%d" fill="url(#hook)"/>' % (top, W, H - top)]
    for i, w in enumerate(words):
        out.append(_t(int(x), 646, w, size, WARM if i == hi else INK, anchor="start"))
        x += widths[i] + space
    return "".join(out)


# ------------------------------------------------------------------ layouts
def _lay_split(d):
    """Two photographs, hard vertical divide, with the things the review said
    were missing: an icon so the eye lands on a shape before a digit, a VS
    badge so the middle is a pivot rather than dead space, and a category
    label so the numbers are not floating free of any subject."""
    half = W // 2
    la, lb = short_name(d["a"]), short_name(d["b"])
    ns = min(fit(la, half - 130, 64, MIN_TEXT), fit(lb, half - 130, 64, MIN_TEXT))
    va, vb = money(d["va"]), money(d["vb"])
    # 520px of number centred at 320 reached x=580, exactly where the VS
    # badge starts. Narrower columns, pushed apart.
    vs = min(fit(va, half - 210, 168, 92), fit(vb, half - 210, 168, 92))
    lab = _context_label(d["subject"])
    ls = fit(lab, 620, 62, MIN_TEXT)
    ico = ICONS[icon_for(d["subject"])]
    up_is_good = bigger_is_better(d["subject"])
    good, bad = (LOW, HIGH) if up_is_good else (HIGH, LOW)

    lx, rx = 300, W - 300
    face_svg, face_w = face_layer(x_right=True, height=244)
    return "".join([
        _defs(),
        _photo_half(d["pa"], 0, half, WARM),
        _photo_half(d["pb"], half, W - half, COOL),
        '<rect x="0" y="0" width="%d" height="540" fill="url(#top)"/>' % W,
        '<rect x="%d" y="0" width="8" height="%d" fill="%s" opacity="0.92"/>' % (half - 4, H, SHADE),
        # context, small but never under the floor
        '<rect x="%d" y="26" width="%d" height="74" rx="10" fill="%s" opacity="0.86"/>'
        % (W // 2 - 330, 660, SHADE),
        ico(W // 2 - text_width(lab, ls) / 2 - 52, 62, 30, WARM),
        _t(W // 2 + 26, 84, lab, ls, WARM),
        # icon, then name, then number - the eye ladder the review asked for
        ico(lx, 170, 62, WARM),
        _t(lx, 258, la, ns, INK),
        _t(lx, 424, va, vs, good if d["va"] >= d["vb"] else bad),
        ico(rx, 170, 62, COOL),
        _t(rx, 258, lb, ns, INK),
        _t(rx, 424, vb, vs, good if d["vb"] > d["va"] else bad),
        _vs_badge(W // 2, 372),
        _hook_band(d["hook"], avoid_right=face_w),
        face_svg,
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
    del _BOXES[:]
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
