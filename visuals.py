"""
visuals.py — the visual engine.

One design language for every graphic in the video: dark navy field, one amber
accent, white type. Everything is drawn as SVG and rasterised with cairosvg, so
there is no browser dependency and no external chart service deciding what our
graphics look like.

Layout rule that governs all of it: the bottom 20% of the frame belongs to the
subtitles. Nothing that carries meaning is drawn there, so captions never sit
on top of a number.

------------------------------------------------------------------------------
v2 — changes made under the QC policy (R002, R006, R011)

R002  bar_pair is now HORIZONTAL and thin. It used to draw two vertical columns
      330px wide and up to ~450px tall, which read as heavy blocks rather than
      a measurement. Bars are now 56px high and up to 1200px long, with the
      figure sitting just past the end of its own bar. Long and thin.

R002  big_stat can carry context_a / context_b. A gap shown on its own is
      meaningless: "$81" tells a viewer nothing without "$1,931 vs $1,850"
      underneath it. The published video showed "$4" and "$81" floating alone.

R011  Three photograph types added — photo_full, photo_stat, photo_split. The
      published video contained zero photographs across 7:47, and one single
      layout occupied 36% of its runtime. Photos come from Pexels (free, no
      per-image cost) and are cached on disk, so a repeated query is fetched
      once. With no PEXELS_API_KEY set they degrade to the normal card rather
      than failing the render.

R006  Every card accepts an "eyebrow": a short category label above the title.
      It is drawn from the spec and never derived from the headline. The old
      cards showed "ADDS UP" sitting above "Every Small Fee Adds Up".

Everything else — the palette, the flag fetching, the US map, the state-shape
fallback, statement, tally, and render_png itself — is unchanged.
------------------------------------------------------------------------------
"""
import os
import json
import math
import base64
import hashlib
import requests
import cairosvg

# ---------------------------------------------------------------- palette
BG      = "#0f172a"
PANEL   = "#1e293b"
INK     = "#f8fafc"
MUTED   = "#94a3b8"
ACCENT  = "#f59e0b"   # side A / the highlighted thing
COOL    = "#3b82f6"   # side B
DIM     = "#334155"   # everything not being talked about

W, H = 1920, 1080
SAFE_H = int(H * 0.80)          # subtitles own everything below this
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"

_HERE = os.path.dirname(os.path.abspath(__file__))
_US_PATHS = None
_FLAG_CACHE = {}
_PHOTO_CACHE = {}
PHOTO_DIR = os.environ.get("PHOTO_CACHE_DIR", "/data/storage/_photos")

# US state and territory postal codes, for flagcdn (us-tx, us-wa ...)
STATE_CODES = {
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
}

COUNTRY_CODES = {
    "united states": "us", "usa": "us", "america": "us",
    "united kingdom": "gb", "uk": "gb", "britain": "gb", "england": "gb",
    "canada": "ca", "australia": "au", "new zealand": "nz", "ireland": "ie",
    "germany": "de", "france": "fr", "spain": "es", "portugal": "pt",
    "italy": "it", "netherlands": "nl", "belgium": "be", "switzerland": "ch",
    "austria": "at", "sweden": "se", "norway": "no", "denmark": "dk",
    "finland": "fi", "poland": "pl", "czechia": "cz", "czech republic": "cz",
    "greece": "gr", "japan": "jp", "south korea": "kr", "korea": "kr",
    "china": "cn", "india": "in", "pakistan": "pk", "singapore": "sg",
    "malaysia": "my", "thailand": "th", "indonesia": "id",
    "united arab emirates": "ae", "uae": "ae", "dubai": "ae",
    "saudi arabia": "sa", "qatar": "qa", "kuwait": "kw", "turkey": "tr",
    "mexico": "mx", "brazil": "br", "argentina": "ar", "chile": "cl",
    "colombia": "co", "south africa": "za", "egypt": "eg", "nigeria": "ng",
    "kenya": "ke", "israel": "il", "russia": "ru", "ukraine": "ua",
}


def _load_paths():
    global _US_PATHS
    if _US_PATHS is None:
        with open(os.path.join(_HERE, "us_paths.json")) as f:
            _US_PATHS = json.load(f)
    return _US_PATHS


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def flag_code(name):
    """Postal code for a US state, ISO code for a country, or None."""
    k = str(name).strip().lower()
    if k in STATE_CODES:
        return "us-" + STATE_CODES[k]
    if k in COUNTRY_CODES:
        return COUNTRY_CODES[k]
    return None


def flag_data_uri(name, width=320):
    """Fetches a flag once and returns it as a data URI.

    Embedding rather than linking matters: cairosvg would otherwise fetch the
    image at rasterise time, and a slow CDN would silently produce a graphic
    with a blank space where the flag should be.
    """
    code = flag_code(name)
    if not code:
        return None
    key = (code, width)
    if key in _FLAG_CACHE:
        return _FLAG_CACHE[key]
    try:
        url = "https://flagcdn.com/w%d/%s.png" % (width, code)
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        uri = "data:image/png;base64," + base64.b64encode(r.content).decode()
        _FLAG_CACHE[key] = uri
        return uri
    except Exception:
        return None


# ==================================================================
#  R011 — photographs
# ==================================================================
def photo_data_uri(query, index=0):
    """One Pexels landscape photo for a query, as a data URI.

    Cached in memory for the run and on disk across runs, so the same metric
    fetches its photograph once no matter how many videos use it. Returns None
    on any failure - a missing photograph degrades the card, it never breaks
    the render.
    """
    if not query:
        return None
    q = str(query).strip().lower()
    # The index is what stops thirty-six photo cards showing the SAME picture.
    # photoQuery() in W3 derives its query from the METRIC, so every scene in a
    # metric section asked for the identical string, and Pexels answered every
    # one of them with its single top result. More than half the video was one
    # photograph with the numbers changing over it.
    #
    # Same query, different rank: scene 1 of a section takes the top photo,
    # scene 2 the second, and so on.
    try:
        idx = max(0, int(index))
    except Exception:
        idx = 0
    key = "%s#%d" % (q, idx)
    if key in _PHOTO_CACHE:
        return _PHOTO_CACHE[key]

    os.makedirs(PHOTO_DIR, exist_ok=True)
    disk = os.path.join(PHOTO_DIR, hashlib.md5(key.encode()).hexdigest() + ".jpg")

    raw = None
    if os.path.exists(disk):
        try:
            with open(disk, "rb") as f:
                raw = f.read()
        except Exception:
            raw = None

    if raw is None:
        # NOT `key` - that name already holds the cache key three lines up.
        # Reusing it overwrote every entry's key with the API key string, so
        # the whole run shared ONE cache slot: the first photograph fetched was
        # handed back for every query and every index after it. This is the
        # second half of the repeated-picture bug, and it would have survived
        # the W3 fix untouched.
        api_key = os.environ.get("PEXELS_API_KEY")
        if not api_key:
            return None
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": q, "per_page": 15,
                        "orientation": "landscape", "size": "large"},
                headers={"Authorization": api_key}, timeout=25)
            r.raise_for_status()
            photos = r.json().get("photos") or []
            if not photos:
                return None
            pick = photos[idx % len(photos)]      # wraps if the search is thin
            src = pick["src"].get("large2x") or pick["src"].get("large")
            img = requests.get(src, timeout=40)
            img.raise_for_status()
            raw = img.content
            try:
                with open(disk, "wb") as f:
                    f.write(raw)
            except Exception:
                pass
        except Exception:
            return None

    uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    _PHOTO_CACHE[key] = uri
    return uri


def _photo_layer(query, x, y, w, h, scrim="bottom", clip_id=None, index=0):
    """A photograph filling a box, with a gradient scrim so type stays legible."""
    uri = photo_data_uri(query, index)
    if not uri:
        return ""
    gid = "sc_%s" % (clip_id or abs(hash((x, y, w, h, scrim))) % 99999)
    if scrim == "bottom":
        grad = ('<linearGradient id="%s" x1="0" y1="0" x2="0" y2="1">'
                '<stop offset="0.30" stop-color="%s" stop-opacity="0.10"/>'
                '<stop offset="1" stop-color="%s" stop-opacity="0.92"/></linearGradient>'
                % (gid, BG, BG))
    elif scrim == "right":
        grad = ('<linearGradient id="%s" x1="0" y1="0" x2="1" y2="0">'
                '<stop offset="0.45" stop-color="%s" stop-opacity="0"/>'
                '<stop offset="1" stop-color="%s" stop-opacity="1"/></linearGradient>'
                % (gid, BG, BG))
    else:  # full — an even wash, for a card that carries a big number on top
        grad = ('<linearGradient id="%s" x1="0" y1="0" x2="0" y2="1">'
                '<stop offset="0" stop-color="%s" stop-opacity="0.72"/>'
                '<stop offset="1" stop-color="%s" stop-opacity="0.86"/></linearGradient>'
                % (gid, BG, BG))
    return ('<defs>%s</defs>'
            '<image x="%.0f" y="%.0f" width="%.0f" height="%.0f" xlink:href="%s" '
            'preserveAspectRatio="xMidYMid slice"/>'
            '<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" fill="url(#%s)"/>'
            % (grad, x, y, w, h, uri, x, y, w, h, gid))


def fit(text, size, max_width, min_size=28):
    """Shrinks a font size until the string fits. DejaVu Sans averages about
    0.58 em per character, which is close enough for headline-length strings."""
    est = len(str(text)) * size * 0.58
    while est > max_width and size > min_size:
        size -= 2
        est = len(str(text)) * size * 0.58
    return size


def wrap(text, chars_per_line, max_lines=4):
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len((cur + " " + w).strip()) <= chars_per_line:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _open(extra=""):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="%d" height="%d" viewBox="0 0 %d %d">'
        '<rect width="%d" height="%d" fill="%s"/>%s'
    ) % (W, H, W, H, W, H, BG, extra)


def _eyebrow(text, y=62):
    """R006: a category label, drawn from the spec. Never taken from the
    headline - "ADDS UP" above "Every Small Fee Adds Up" reads as a bug."""
    if not text:
        return ""
    return ('<text x="%d" y="%d" fill="%s" font-family="%s" font-size="30" '
            'font-weight="bold" text-anchor="middle" letter-spacing="5">%s</text>'
            ) % (W // 2, y, ACCENT, FONT, esc(str(text).upper()))


def _title(text, y=110):
    if not text:
        return ""
    size = fit(text, 52, W - 280, 34)
    return ('<text x="%d" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle" letter-spacing="1">%s</text>'
            ) % (W // 2, y, MUTED, FONT, size, esc(text.upper()))


# ==================================================================
#  1. flag_vs — two places, side by side
# ==================================================================
def flag_vs(a, b, title=None, sub_a=None, sub_b=None, eyebrow=None):
    parts = [_eyebrow(eyebrow), _title(title, 130)]
    fw, fh = 560, 373
    cy = 420
    for i, (name, sub, colour) in enumerate([(a, sub_a, ACCENT), (b, sub_b, COOL)]):
        cx = W * (0.27 if i == 0 else 0.73)
        uri = flag_data_uri(name, 640)
        x, y = cx - fw / 2, cy - fh / 2
        if uri:
            parts.append(
                '<image x="%d" y="%d" width="%d" height="%d" xlink:href="%s" '
                'preserveAspectRatio="xMidYMid slice"/>'
                '<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="%s" stroke-width="5"/>'
                % (x, y, fw, fh, uri, x, y, fw, fh, colour))
        else:
            parts.append(state_shape_svg(name, fw, fh, x, y, colour))
        size = fit(name, 76, fw + 140, 38)
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, cy + fh / 2 + 108, INK, FONT, size, esc(name)))
        if sub:
            # These carry the hook figure on the title card, so they are set as
            # figures - in the entity's own colour, at a size a viewer reads in
            # the first second - not as grey small print under the name.
            parts.append(
                '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="72" '
                'font-weight="bold" text-anchor="middle">%s</text>'
                % (cx, cy + fh / 2 + 184, colour, FONT, esc(sub)))
    parts.append(
        '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="62" '
        'font-weight="bold" text-anchor="middle">vs</text>'
        % (W // 2, cy + 22, MUTED, FONT))
    return _open() + "".join(parts) + "</svg>"


def state_shape_svg(name, box_w, box_h, x, y, colour):
    """Fallback when a flag image is unavailable: draw the place's own outline.

    A grey rectangle where a flag should be looks broken. The state silhouette
    carries the same information and uses geometry we already ship.
    """
    paths = _load_paths()
    key = next((k for k in paths if k.lower() == str(name).strip().lower()), None)
    if not key:
        return ('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" '
                'stroke="%s" stroke-width="4"/>' % (x, y, box_w, box_h, PANEL, colour))
    d = paths[key]
    xs, ys = [], []
    for tok in d.replace("M", " ").replace("L", " ").replace("Z", " ").split():
        if "," in tok:
            px, py = tok.split(",")
            xs.append(float(px)); ys.append(float(py))
    if not xs:
        return ""
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    sc = min(box_w / (bw or 1), box_h / (bh or 1)) * 0.86
    tx = x + (box_w - bw * sc) / 2 - min(xs) * sc
    ty = y + (box_h - bh * sc) / 2 - min(ys) * sc
    return ('<g transform="translate(%.1f,%.1f) scale(%.3f)">'
            '<path d="%s" fill="%s" stroke="%s" stroke-width="%.1f"/></g>'
            % (tx, ty, sc, d, colour, INK, 2.0 / sc))


# ==================================================================
#  2. map — one or two states located on the country
# ==================================================================
def us_map(highlight, title=None, labels=True, eyebrow=None):
    if isinstance(highlight, str):
        highlight = [highlight]
    hl = {h.strip().lower(): i for i, h in enumerate(highlight)}
    colours = [ACCENT, COOL]
    paths = _load_paths()
    body, marks = [], []

    # The map art is 1000x620; place it centred inside the safe area.
    mw, mh = 1000, 620
    sc = 1.42
    ox = (W - mw * sc) / 2
    oy = 150

    for name, d in paths.items():
        idx = hl.get(name.lower())
        fill = colours[idx % 2] if idx is not None else DIM
        stroke = INK if idx is not None else BG
        body.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%s"/>'
                    % (d, fill, stroke, "2.4" if idx is not None else "1.4"))
    g = ('<g transform="translate(%.1f,%.1f) scale(%.3f)">%s</g>'
         % (ox, oy, sc, "".join(body)))

    out = [_eyebrow(eyebrow), _title(title, 100), g]

    if labels:
        for i, name in enumerate(highlight):
            key = next((k for k in paths if k.lower() == name.strip().lower()), None)
            if not key:
                continue
            d = paths[key]
            nums = [p for p in d.replace("M", " ").replace("L", " ").replace("Z", " ").split()]
            xs, ys = [], []
            for n in nums:
                if "," in n:
                    px, py = n.split(",")
                    xs.append(float(px)); ys.append(float(py))
            if not xs:
                continue
            cx = ox + (sum(xs) / len(xs)) * sc
            cy = oy + (sum(ys) / len(ys)) * sc
            size = fit(name, 46, 460, 30)
            tw = len(name) * size * 0.60 + 44
            marks.append(
                '<rect x="%.0f" y="%.0f" width="%.0f" height="62" rx="8" fill="%s" opacity="0.92"/>'
                '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
                'font-weight="bold" text-anchor="middle">%s</text>'
                % (cx - tw / 2, cy - 31, tw, BG, cx, cy + 15, colours[i % 2], FONT, size, esc(name)))
        out.extend(marks)
    return _open() + "".join(out) + "</svg>"


# ==================================================================
#  3. bar_pair — the workhorse: two figures, one metric
# ==================================================================
def bar_pair(label_a, value_a, label_b, value_b, title=None,
             display_a=None, display_b=None, unit=None,
             eyebrow=None, photo_query=None, photo_index=0):
    """
    R002 — HORIZONTAL and thin.

    This used to draw two vertical columns 330px wide and up to ~450px tall.
    They read as heavy blocks rather than a measurement, and the two figures
    were the only thing on screen for up to fifty seconds at a time.

    Now: 56px high, up to 1200px long, label in its own column on the left,
    figure just past the end of its own bar, faint gridlines behind. Long and
    thin. If photo_query is given the left third carries a photograph and the
    bars occupy the rest.
    """
    va, vb = float(value_a), float(value_b)
    top = max(abs(va), abs(vb)) or 1.0
    da = display_a if display_a is not None else "{:,.0f}".format(va)
    db = display_b if display_b is not None else "{:,.0f}".format(vb)

    parts = []
    bar_x, max_len = 460, 1160

    if photo_query:
        pw = int(W * 0.34)
        layer = _photo_layer(photo_query, 0, 0, pw, SAFE_H, scrim="right", clip_id="bp",
                             index=photo_index)
        if layer:
            parts.append(layer)
            bar_x, max_len = pw + 300, W - (pw + 300) - 260

    parts.append(_eyebrow(eyebrow))
    parts.append(_title(title, 130))

    BAR_H, BAR_GAP = 56, 40
    block_h = BAR_H * 2 + BAR_GAP
    top_y = 470 - block_h // 2

    # faint gridlines behind the bars, so a length reads as a quantity
    for g in (0.25, 0.5, 0.75):
        gx = bar_x + max_len * g
        parts.append('<line x1="%.0f" y1="%.0f" x2="%.0f" y2="%.0f" stroke="%s" '
                     'stroke-width="2"/>' % (gx, top_y - 30, gx, top_y + block_h + 30, PANEL))

    for i, (lab, val, disp, colour) in enumerate(
            [(label_a, va, da, ACCENT), (label_b, vb, db, COOL)]):
        by = top_y + i * (BAR_H + BAR_GAP)
        length = max(10, (abs(val) / top) * max_len)

        # label right-aligned into its own column, never on top of the bar
        lsize = fit(lab, 44, bar_x - 60, 26)
        parts.append(
            '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="end">%s</text>'
            % (bar_x - 34, by + BAR_H * 0.72, MUTED, FONT, lsize, esc(lab)))

        parts.append('<rect x="%.0f" y="%.0f" width="%.0f" height="%d" rx="4" fill="%s"/>'
                     % (bar_x, by, length, BAR_H, colour))

        # figure just past the end of its own bar - outside, never inside
        vsize = 52
        vx = bar_x + length + 26
        if vx + len(str(disp)) * vsize * 0.58 > W - 40:
            vx = bar_x + length - 26
            parts.append(
                '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
                'font-weight="bold" text-anchor="end">%s</text>'
                % (vx, by + BAR_H * 0.74, BG, FONT, vsize, esc(disp)))
        else:
            parts.append(
                '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
                'font-weight="bold">%s</text>'
                % (vx, by + BAR_H * 0.74, INK, FONT, vsize, esc(disp)))

    if unit:
        parts.append(
            '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="34" '
            'text-anchor="middle">%s</text>'
            % (bar_x + max_len / 2, top_y + block_h + 110, MUTED, FONT, esc(unit)))

    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  4. big_stat — one figure, full frame
# ==================================================================
def big_stat(value, caption=None, title=None, colour=ACCENT,
             eyebrow=None, context_a=None, context_b=None, photo_query=None,
             photo_index=0):
    """
    R002 — a gap is never shown on its own.

    The published video put "$4" and "$81" alone on a dark field. Eighty-one
    dollars means nothing without the figures it came from; against rent of
    $1,850 it is 4.4%. When context_a / context_b are supplied they are drawn
    under the headline number.
    """
    parts = []

    # Did we actually get a photograph? This matters more than it looks.
    #
    # photo_stat and big_stat are the same function; only photo_query differs.
    # When the photo could not be fetched the two rendered BYTE-IDENTICAL, so a
    # sequence built to alternate between them showed one unchanging picture -
    # measured as an 18.7-second static hold, with the cuts between the pieces
    # invisible to scene detection.
    #
    # A missing photo must still produce a different card, so the layout shifts:
    # the figure moves off-centre against an accent rule instead of sitting
    # dead centre. Nothing depends on the network to stay visually distinct.
    layer = None
    if photo_query:
        layer = _photo_layer(photo_query, 0, 0, W, H, scrim="full", clip_id="bs",
                             index=photo_index)
        if layer:
            parts.append(layer)
    # _photo_layer returns "" when there is no photo, not None. Testing for
    # None meant the fallback layout never triggered and the two cards stayed
    # byte-identical - the bug this whole branch exists to prevent.
    offset = bool(photo_query) and not layer

    parts.append(_eyebrow(eyebrow))
    parts.append(_title(title, 130))

    size = fit(value, 260, W - 320, 90)
    if offset:
        parts.append('<rect x="140" y="368" width="12" height="210" fill="%s" rx="6"/>'
                     % colour)
        parts.append(
            '<text x="196" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold">%s</text>'
            % (540, colour, FONT, min(size, 230), esc(value)))
    else:
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (W // 2, 520, colour, FONT, size, esc(value)))

    y = 640
    if caption:
        for i, ln in enumerate(wrap(caption, 34, 2)):
            if offset:
                parts.append(
                    '<text x="196" y="%d" fill="%s" font-family="%s" font-size="52">%s</text>'
                    % (y + i * 74, INK, FONT, esc(ln)))
            else:
                parts.append(
                    '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="56" '
                    'text-anchor="middle">%s</text>'
                    % (W // 2, y + i * 74, INK, FONT, esc(ln)))
        y += 74 * len(wrap(caption, 34, 2))

    if context_a and context_b:
        ay = min(y + 46, SAFE_H - 40)
        a_txt = "%s  %s" % (context_a.get("label", ""), context_a.get("display", ""))
        b_txt = "%s  %s" % (context_b.get("label", ""), context_b.get("display", ""))
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="40" '
            'font-weight="bold" text-anchor="end">%s</text>'
            % (W // 2 - 46, ay, ACCENT, FONT, esc(a_txt)))
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="32" '
            'text-anchor="middle">vs</text>' % (W // 2, ay, MUTED, FONT))
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="40" '
            'font-weight="bold">%s</text>'
            % (W // 2 + 46, ay, COOL, FONT, esc(b_txt)))

    if not offset:
        parts.append('<rect x="%d" y="%d" width="220" height="8" fill="%s" rx="4"/>'
                     % (W // 2 - 110, 300, colour))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  5. photo_full — a photograph carrying one line of caption
# ==================================================================
def photo_full(photo_query, caption=None, eyebrow=None, title=None, photo_index=0):
    """R011. The published video had no photographs at all, and 36% of its
    runtime was a single repeated layout. A real image every few shots is the
    cheapest variety there is."""
    layer = _photo_layer(photo_query, 0, 0, W, H, scrim="bottom", clip_id="pf",
                         index=photo_index)
    if not layer:
        # No key, no network, no photo - fall back to a card that still reads.
        return statement(caption or title or "", kicker=eyebrow)

    parts = [layer, _eyebrow(eyebrow)]
    if caption:
        lines = wrap(caption, 32, 3)
        size = 72 if len(lines) <= 2 else 60
        start = SAFE_H - 40 - (len(lines) - 1) * size * 1.24
        for i, ln in enumerate(lines):
            parts.append(
                '<text x="%d" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
                'font-weight="bold" text-anchor="middle">%s</text>'
                % (W // 2, start + i * size * 1.24, INK, FONT, size, esc(ln)))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  6. statement — typographic card, no data
# ==================================================================
def statement(text, kicker=None, eyebrow=None):
    parts = []
    lead = kicker or eyebrow
    if lead:
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="40" '
            'text-anchor="middle" letter-spacing="3">%s</text>'
            % (W // 2, 210, ACCENT, FONT, esc(str(lead).upper())))
    lines = wrap(text, 26, 4)
    size = 96 if len(lines) <= 2 else 76
    start = 420 - (len(lines) - 1) * (size * 0.62)
    for i, ln in enumerate(lines):
        parts.append(
            '<text x="%d" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (W // 2, start + i * size * 1.24, INK, FONT, size, esc(ln)))
    parts.append('<rect x="%d" y="%d" width="160" height="6" fill="%s" rx="3"/>'
                 % (W // 2 - 80, 280, ACCENT))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  7. tally — the running scoreboard
# ==================================================================
def tally(name_a, score_a, name_b, score_b, title="RUNNING TOTAL", rows=None,
          eyebrow=None):
    """
    R007 — the detail list is left-aligned in one column with its dot in a
    fixed gutter. It used to be centred with the dot placed under whichever
    side won, which put the marker on top of the text: "...car insurance per
    yea●". The list also used to run past the safe line into the subtitles.
    """
    parts = [_eyebrow(eyebrow), _title(title, 110)]
    cy = 300
    for i, (name, score, colour) in enumerate(
            [(name_a, score_a, ACCENT), (name_b, score_b, COOL)]):
        cx = W * (0.28 if i == 0 else 0.72)
        parts.append(
            '<text x="%.0f" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, cy, MUTED, FONT, fit(name, 52, 620, 32), esc(name)))
        parts.append(
            '<text x="%.0f" y="%d" fill="%s" font-family="%s" font-size="168" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, cy + 150, colour, FONT, esc(score)))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                 % (W // 2, cy - 60, W // 2, cy + 172, PANEL))

    if rows:
        list_x = 520          # fixed gutter for the dots
        text_x = list_x + 46
        y = cy + 246
        room = max(0, (SAFE_H - 24 - y) // 46)
        for r in rows[:min(6, room)]:
            who = str(r.get("winner", ""))
            left = who.strip().lower() == str(name_a).strip().lower()
            colour = ACCENT if left else COOL
            metric = str(r.get("metric", ""))
            if len(metric) > 42:
                metric = metric[:41].rstrip() + "…"
            parts.append(
                '<circle cx="%d" cy="%d" r="11" fill="%s"/>'
                '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="36">%s</text>'
                % (list_x, y - 11, colour, text_x, y, MUTED, FONT, esc(metric)))
            y += 46
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  dispatcher
# ==================================================================
def build_svg(spec):
    t = str(spec.get("type", "statement")).lower()
    eb = spec.get("eyebrow")

    if t == "flag_vs":
        return flag_vs(spec.get("entity_a"), spec.get("entity_b"),
                       spec.get("title"), spec.get("sub_a"), spec.get("sub_b"),
                       eyebrow=eb)

    if t == "map":
        return us_map(spec.get("highlight") or [], spec.get("title"), eyebrow=eb)

    if t in ("bar_pair", "photo_split"):
        return bar_pair(spec.get("label_a"), spec.get("value_a"),
                        spec.get("label_b"), spec.get("value_b"),
                        spec.get("title"), spec.get("display_a"),
                        spec.get("display_b"), spec.get("unit"),
                        eyebrow=eb,
                        photo_query=spec.get("photo_query") if t == "photo_split" else None,
                        photo_index=spec.get("photo_index", 0))

    if t in ("big_stat", "photo_stat"):
        return big_stat(spec.get("value", ""), spec.get("caption"), spec.get("title"),
                        eyebrow=eb,
                        context_a=spec.get("context_a"), context_b=spec.get("context_b"),
                        photo_query=spec.get("photo_query") if t == "photo_stat" else None,
                        photo_index=spec.get("photo_index", 0))

    if t == "photo_full":
        return photo_full(spec.get("photo_query"), spec.get("caption"),
                          eyebrow=eb, title=spec.get("title"),
                          photo_index=spec.get("photo_index", 0))

    if t == "tally":
        return tally(spec.get("name_a"), spec.get("score_a"),
                     spec.get("name_b"), spec.get("score_b"),
                     spec.get("title", "RUNNING TOTAL"), spec.get("rows"),
                     eyebrow=eb)

    return statement(spec.get("text", ""), spec.get("kicker"), eyebrow=eb)


def render_png(spec, dest_path):
    svg = build_svg(spec)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=dest_path,
                     output_width=W, output_height=H)
    return dest_path
