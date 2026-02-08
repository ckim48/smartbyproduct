import os
import json
import sqlite3
from contextlib import closing
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)


client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DB_PATH = os.getenv("INVENTORY_DB_PATH", "inventory.db")
EPS = 1e-6


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                qty REAL NOT NULL,
                unit TEXT NOT NULL,
                minor_part TEXT NOT NULL DEFAULT 'Unknown',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL DEFAULT (datetime('now')),
                item_id INTEGER,
                item_name TEXT,
                unit TEXT,
                qty REAL,
                details_json TEXT
            )
            """
        )
        conn.commit()


def fetch_inventory():
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT id, name, qty, unit, minor_part FROM inventory ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def normalize_name(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def parse_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def log_event(event_type: str, *, item_id=None, item_name=None, unit=None, qty=None, details=None):
    try:
        details_json = json.dumps(details, ensure_ascii=False) if details is not None else None
    except Exception:
        details_json = None

    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO logs (event_type, item_id, item_name, unit, qty, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_type, item_id, item_name, unit, qty, details_json),
        )
        conn.commit()


def fetch_logs(limit=200):
    limit = int(limit) if str(limit).isdigit() else 200
    limit = max(1, min(limit, 1000))

    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, event_type, event_time, item_id, item_name, unit, qty, details_json
            FROM logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = json.loads(d["details_json"]) if d.get("details_json") else None
            except Exception:
                d["details"] = None
            d.pop("details_json", None)
            out.append(d)
        return out


def get_minor_parts(food_item: str) -> str:
    if not client:
        return "Unknown"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a food waste expert. For any grocery item given, list its common edible "
                        "or non-edible byproducts (e.g., peel, core, skin, shell, seeds, tops, stems). "
                        "Return a short comma-separated list. If none, return 'None'. Do not use emojis."
                    ),
                },
                {"role": "user", "content": f"Item: {food_item}\nReturn only the comma-separated list."},
            ],
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else "None"
    except Exception:
        return "Unknown"


def generate_recipe_preview(inv_items):
    if not client:
        return {
            "title": "",
            "steps": [],
            "recipe_text": "Recipe generation is unavailable because OPENAI_API_KEY is not set.",
            "used": [],
        }

    if not inv_items:
        return {
            "title": "",
            "steps": [],
            "recipe_text": "Inventory is empty. Add at least one item to generate a recipe.",
            "used": [],
        }

    lines = []
    for it in inv_items:
        lines.append(
            f"- id={it['id']} name={it['name']} available={it['qty']} unit={it['unit']} byproducts={it.get('minor_part','Unknown')}"
        )
    inv_text = "\n".join(lines)

    schema = """
Return ONLY valid JSON in this exact format:
{
  "title": "string",
  "steps": ["string", "string"],
  "used": [
    {"id": number, "qty": number}
  ]
}

Rules:
- title: short food name (3 to 8 words). No emojis.
- steps: a list of clear cooking steps (4 to 8 steps). Each step is one sentence. No emojis.
- used: ONLY use ids from the inventory list.
- Never include byproducts (peel, seeds, shells) as separate used items.
- Do not repeat the same id multiple times.
- qty must be <= available for that id.
- Use 1 to 4 inventory items only.
- qty must be realistic and positive.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a practical home-cooking assistant focused on food waste reduction. "
                        "You must output JSON only and follow the schema strictly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Inventory:\n{inv_text}\n\n"
                        "Generate exactly ONE simple recipe that uses byproducts as much as possible (when safe). "
                        "Byproducts are derived from the main items; do not treat them as separate inventory items.\n\n"
                        + schema
                    ),
                },
            ],
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON object.")

        obj = json.loads(raw[start : end + 1])

        title = str(obj.get("title") or "").strip()
        steps = obj.get("steps") or []
        used = obj.get("used") or []

        if not isinstance(steps, list):
            steps = []

        cleaned_steps = []
        for s in steps:
            if not isinstance(s, str):
                continue
            t = " ".join(s.strip().split())
            t = t.lstrip("-•*0123456789. )(").strip()
            if t:
                cleaned_steps.append(t)
        cleaned_steps = cleaned_steps[:12]

        if not title:
            title = "Simple Waste-Reducing Recipe"
        title = " ".join(title.split())[:80]

        inv_map = {int(it["id"]): it for it in inv_items}
        dedup = {}
        if isinstance(used, list):
            for u in used:
                if not isinstance(u, dict):
                    continue
                iid = u.get("id")
                qty = parse_float(u.get("qty"), None)
                if iid is None or qty is None:
                    continue
                try:
                    iid = int(iid)
                except Exception:
                    continue
                if iid not in inv_map:
                    continue
                if qty <= 0:
                    continue

                avail = float(inv_map[iid]["qty"])
                qty = min(qty, avail)
                qty = round(qty, 2)
                if iid not in dedup or qty > dedup[iid]:
                    dedup[iid] = qty

        cleaned_used = [{"id": iid, "qty": dedup[iid]} for iid in dedup.keys()]

        recipe_text = title
        if cleaned_steps:
            recipe_text += "\n" + "\n".join([f"{i+1}. {cleaned_steps[i]}" for i in range(len(cleaned_steps))])

        return {"title": title, "steps": cleaned_steps, "recipe_text": recipe_text, "used": cleaned_used}

    except Exception:
        return {"title": "", "steps": [], "recipe_text": "Sorry—recipe generation failed. Please try again.", "used": []}


@app.route("/")
def index():
    return render_template("index.html", inventory=fetch_inventory())


@app.route("/inventory")
def inventory_page():
    return render_template("index.html", inventory=fetch_inventory())


@app.post("/api/minor_parts")
def api_minor_parts():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Missing item name"}), 400
    return jsonify({"ok": True, "minor_part": get_minor_parts(name)})


@app.post("/api/add_item")
def api_add_item():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    unit = (data.get("unit") or "pcs").strip() or "pcs"
    minor_part = (data.get("minor_part") or "Unknown").strip()

    if not name:
        return jsonify({"ok": False, "error": "Missing item name"}), 400

    qty = parse_float(data.get("qty", 1), None)
    if qty is None or qty <= 0:
        return jsonify({"ok": False, "error": "Quantity must be a positive number"}), 400

    nkey = normalize_name(name)

    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT id, name, qty, unit, minor_part FROM inventory WHERE lower(trim(name))=? AND unit=? LIMIT 1",
            (nkey, unit),
        ).fetchone()

        if row:
            before = float(row["qty"])
            new_qty = round(before + qty, 2)
            new_minor = row["minor_part"]
            if (new_minor in ("Unknown", "", None)) and (minor_part not in ("Unknown", "", None)):
                new_minor = minor_part

            conn.execute(
                "UPDATE inventory SET qty=?, minor_part=? WHERE id=?",
                (new_qty, new_minor, row["id"]),
            )
            conn.commit()

            log_event(
                "ADD",
                item_id=int(row["id"]),
                item_name=row["name"],
                unit=row["unit"],
                qty=round(qty, 2),
                details={"mode": "merge", "before": round(before, 2), "after": round(new_qty, 2), "minor_part": new_minor},
            )
            return jsonify({"ok": True})

        cur = conn.execute(
            "INSERT INTO inventory (name, qty, unit, minor_part) VALUES (?, ?, ?, ?)",
            (name, round(qty, 2), unit, minor_part or "Unknown"),
        )
        new_id = cur.lastrowid
        conn.commit()

    log_event(
        "ADD",
        item_id=int(new_id),
        item_name=name,
        unit=unit,
        qty=round(qty, 2),
        details={"mode": "new", "minor_part": minor_part or "Unknown"},
    )
    return jsonify({"ok": True})


@app.post("/api/delete_item")
def api_delete_item():
    data = request.get_json(silent=True) or {}
    try:
        item_id = int(data.get("id"))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid id"}), 400

    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT id, name, qty, unit, minor_part FROM inventory WHERE id=?",
            (item_id,),
        ).fetchone()

        cur = conn.execute("DELETE FROM inventory WHERE id=?", (item_id,))
        conn.commit()

    if cur.rowcount == 0:
        return jsonify({"ok": False, "error": "Item not found"}), 404

    if row:
        log_event(
            "DELETE",
            item_id=int(row["id"]),
            item_name=row["name"],
            unit=row["unit"],
            qty=round(float(row["qty"]), 2),
            details={"minor_part": row["minor_part"]},
        )

    return jsonify({"ok": True})


@app.post("/api/inventory")
def api_inventory():
    return jsonify({"ok": True, "inventory": fetch_inventory()})


@app.post("/api/recipe_preview")
def api_recipe_preview():
    inv = fetch_inventory()
    result = generate_recipe_preview(inv)
    return jsonify(
        {
            "ok": True,
            "title": result.get("title") or "",
            "steps": result.get("steps") or [],
            "recipe_text": result.get("recipe_text") or "",
            "used": result.get("used") or [],
        }
    )


@app.post("/api/make_recipe")
def api_make_recipe():
    data = request.get_json(silent=True) or {}
    used = data.get("used") or []
    recipe_text = (data.get("recipe_text") or "").strip()
    title = (data.get("title") or "").strip()
    steps = data.get("steps") or []

    if not isinstance(used, list) or not used:
        return jsonify({"ok": False, "error": "No used-items provided."}), 400

    dedup = {}
    for u in used:
        if not isinstance(u, dict):
            continue
        iid = u.get("id")
        qty = parse_float(u.get("qty"), None)
        if iid is None or qty is None or qty <= 0:
            continue
        try:
            iid = int(iid)
        except Exception:
            continue
        qty = round(qty, 2)
        if iid not in dedup or qty > dedup[iid]:
            dedup[iid] = qty

    if not dedup:
        return jsonify({"ok": False, "error": "Invalid used-items payload."}), 400

    applied = []
    used_summary = []

    with closing(get_db()) as conn:
        for iid, sub_qty in dedup.items():
            row = conn.execute("SELECT id, name, qty, unit FROM inventory WHERE id=?", (iid,)).fetchone()
            if not row:
                return jsonify({"ok": False, "error": f"Item id {iid} not found in inventory."}), 400

            avail = float(row["qty"])
            if sub_qty > avail + EPS:
                return jsonify({"ok": False, "error": f"Not enough quantity for {row['name']} ({row['unit']}). Need {sub_qty}, have {avail}."}), 400

        for iid, sub_qty in dedup.items():
            row = conn.execute("SELECT id, name, qty, unit FROM inventory WHERE id=?", (iid,)).fetchone()
            avail = float(row["qty"])
            new_qty = round(avail - sub_qty, 2)

            if new_qty <= EPS:
                conn.execute("DELETE FROM inventory WHERE id=?", (iid,))
                new_qty = 0.0
            else:
                conn.execute("UPDATE inventory SET qty=? WHERE id=?", (new_qty, iid))

            applied.append({
                "id": iid,
                "name": row["name"],
                "unit": row["unit"],
                "before": round(avail, 2),
                "used": round(sub_qty, 2),
                "after": round(new_qty, 2),
            })

            used_summary.append({
                "id": iid,
                "name": row["name"],
                "qty": round(sub_qty, 2),
                "unit": row["unit"],
            })

        conn.commit()

    if (not recipe_text) and title and isinstance(steps, list) and steps:
        safe_steps = []
        for s in steps:
            if isinstance(s, str) and s.strip():
                safe_steps.append(" ".join(s.strip().split()))
        recipe_text = title + "\n" + "\n".join([f"{i+1}. {safe_steps[i]}" for i in range(len(safe_steps))])

    log_event(
        "MAKE_RECIPE",
        details={
            "title": title if title else None,
            "steps": steps if isinstance(steps, list) else None,
            "recipe_text": recipe_text if recipe_text else "(no recipe_text provided)",
            "used": used_summary,
            "applied": applied,
        },
    )

    return jsonify({"ok": True, "inventory": fetch_inventory(), "applied": applied})


@app.post("/api/logs")
def api_logs():
    data = request.get_json(silent=True) or {}
    limit = data.get("limit", 200)
    return jsonify({"ok": True, "logs": fetch_logs(limit=limit)})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
