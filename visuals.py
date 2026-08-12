"""
visuals.py — the visual engine.

One design language for every graphic in the video: dark navy field, one amber
accent, white type. Everything is drawn as SVG and rasterised with cairosvg, so
there is no browser dependency and no external chart service deciding what our
graphics look like.

Layout rule that governs all of it: the bottom 20% of the frame belongs to the
subtitles. Nothing that carries meaning is drawn there, so captions never sit
on top of a number.
"""

import os
import json
import math
import base64
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
def flag_vs(a, b, title=None, sub_a=None, sub_b=None):
    parts = [_title(title, 130)]
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
            parts.append(
                '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="42" '
                'text-anchor="middle">%s</text>'
                % (cx, cy + fh / 2 + 172, MUTED, FONT, esc(sub)))

    parts.append(
        '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="62" '
        'font-weight="bold" text-anchor="middle">vs</text>'
        % (W // 2, cy + 22, MUTED, FONT))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  2. map — one or two states located on the country
# ==================================================================
def us_map(highlight, title=None, labels=True):
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

    out = [_title(title, 100), g]

    if labels:
        for i, name in enumerate(highlight):
            key = next((k for k in paths if k.lower() == name.strip().lower()), None)
            if not key:
                continue
            # Label position: centroid of the largest ring, roughly.
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
             display_a=None, display_b=None, unit=None):
    va, vb = float(value_a), float(value_b)
    top = max(va, vb) or 1.0
    da = display_a or ("%,d" % int(va)).replace("%", "")
    db = display_b or ("%,d" % int(vb)).replace("%", "")

    parts = [_title(title, 110)]

    base_y = SAFE_H - 130          # bars sit on this line
    max_bar = base_y - 300         # tallest bar height
    bw = 330

    for i, (lab, val, disp, colour) in enumerate(
            [(label_a, va, da, ACCENT), (label_b, vb, db, COOL)]):
        cx = W * (0.30 if i == 0 else 0.70)
        bh = max(60, (val / top) * max_bar)
        x, y = cx - bw / 2, base_y - bh
        parts.append('<rect x="%.0f" y="%.0f" width="%d" height="%.0f" fill="%s" rx="6"/>'
                     % (x, y, bw, bh, colour))
        # figure sits above its own bar - never in a legend, never in a corner
        size = fit(disp, 78, bw + 200, 40)
        parts.append(
            '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, y - 34, INK, FONT, size, esc(disp)))
        lsize = fit(lab, 50, bw + 190, 30)
        parts.append(
            '<text x="%.0f" y="%.0f" fill="%s" font-family="%s" font-size="%d" '
            'text-anchor="middle">%s</text>'
            % (cx, base_y + 62, MUTED, FONT, lsize, esc(lab)))

    parts.append('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="%s" stroke-width="3"/>'
                 % (200, base_y, W - 200, base_y, PANEL))
    if unit:
        parts.append(
            '<text x="%d" y="%.0f" fill="%s" font-family="%s" font-size="34" '
            'text-anchor="middle">%s</text>'
            % (W // 2, base_y + 130, MUTED, FONT, esc(unit)))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  4. big_stat — one figure, full frame
# ==================================================================
def big_stat(value, caption=None, title=None, colour=ACCENT):
    parts = [_title(title, 130)]
    size = fit(value, 260, W - 320, 90)
    parts.append(
        '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="%d" '
        'font-weight="bold" text-anchor="middle">%s</text>'
        % (W // 2, 520, colour, FONT, size, esc(value)))
    if caption:
        lines = wrap(caption, 34, 2)
        for i, ln in enumerate(lines):
            parts.append(
                '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="56" '
                'text-anchor="middle">%s</text>'
                % (W // 2, 640 + i * 74, INK, FONT, esc(ln)))
    parts.append('<rect x="%d" y="%d" width="220" height="8" fill="%s" rx="4"/>'
                 % (W // 2 - 110, 300, colour))
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  5. statement — typographic card, no data
# ==================================================================
def statement(text, kicker=None):
    parts = []
    if kicker:
        parts.append(
            '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="40" '
            'text-anchor="middle" letter-spacing="3">%s</text>'
            % (W // 2, 210, ACCENT, FONT, esc(kicker.upper())))
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
#  6. tally — the running scoreboard
# ==================================================================
def tally(name_a, score_a, name_b, score_b, title="RUNNING TOTAL", rows=None):
    parts = [_title(title, 110)]
    cy = 330
    for i, (name, score, colour) in enumerate(
            [(name_a, score_a, ACCENT), (name_b, score_b, COOL)]):
        cx = W * (0.28 if i == 0 else 0.72)
        parts.append(
            '<text x="%.0f" y="%d" fill="%s" font-family="%s" font-size="%d" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, cy, MUTED, FONT, fit(name, 52, 620, 32), esc(name)))
        parts.append(
            '<text x="%.0f" y="%d" fill="%s" font-family="%s" font-size="200" '
            'font-weight="bold" text-anchor="middle">%s</text>'
            % (cx, cy + 175, colour, FONT, esc(score)))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                 % (W // 2, cy - 60, W // 2, cy + 200, PANEL))

    # optional detail list: which metric each side took
    if rows:
        y = cy + 300
        for r in rows[:6]:
            who = str(r.get("winner", ""))
            left = who.strip().lower() == str(name_a).strip().lower()
            colour = ACCENT if left else COOL
            dot_x = W * (0.28 if left else 0.72)
            parts.append(
                '<text x="%d" y="%d" fill="%s" font-family="%s" font-size="38" '
                'text-anchor="middle">%s</text>'
                '<circle cx="%.0f" cy="%d" r="13" fill="%s"/>'
                % (W // 2, y, MUTED, FONT, esc(r.get("metric", "")),
                   dot_x, y - 12, colour))
            y += 58
    return _open() + "".join(parts) + "</svg>"


# ==================================================================
#  dispatcher
# ==================================================================
def build_svg(spec):
    t = str(spec.get("type", "statement")).lower()
    if t == "flag_vs":
        return flag_vs(spec.get("entity_a"), spec.get("entity_b"),
                       spec.get("title"), spec.get("sub_a"), spec.get("sub_b"))
    if t == "map":
        return us_map(spec.get("highlight") or [], spec.get("title"))
    if t == "bar_pair":
        return bar_pair(spec.get("label_a"), spec.get("value_a"),
                        spec.get("label_b"), spec.get("value_b"),
                        spec.get("title"), spec.get("display_a"),
                        spec.get("display_b"), spec.get("unit"))
    if t == "big_stat":
        return big_stat(spec.get("value", ""), spec.get("caption"), spec.get("title"))
    if t == "tally":
        return tally(spec.get("name_a"), spec.get("score_a"),
                     spec.get("name_b"), spec.get("score_b"),
                     spec.get("title", "RUNNING TOTAL"), spec.get("rows"))
    return statement(spec.get("text", ""), spec.get("kicker"))


def render_png(spec, dest_path):
    svg = build_svg(spec)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=dest_path,
                     output_width=W, output_height=H)
    return dest_path
