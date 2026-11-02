"""
חילוץ מתכונים ממסמכי המאגר (DOCX ו־TXT) לקובץ מובנה data/recipes.json.
זהו שלב מקדים: האינדקס נבנה מ־recipes.json בקובץ index_recipes.py.

    python extract_recipes.py           חילוץ המתכונים ויצירת extraction_report.txt
    python extract_recipes.py --force   חילוץ מחדש, גם אם recipes.json כבר קיים (דורס תיקונים ידניים!)
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

DATA_DIR = Path(__file__).parent / "data"
RECIPES_FILE = DATA_DIR / "recipes.json"
REPORT_FILE = Path(__file__).parent / "extraction_report.txt"

# מילים שפותחות כותרת של חלק בתוך מתכון (בצק, מילוי וכו'), ולא כותרת של מתכון חדש
SECTION_WORDS = [
    "חומרים", "רכיבים", "מצרכים", "אופן ההכנה", "אופן הכנה", "הכנה", "הוראות",
    "בצק", "מילוי", "מלית", "ציפוי", "קרם", "רוטב", "בסיס", "שכבה", "שכבת",
    "תיבול", "התיבול", "תחמיץ", "בלילה", "קישוט", "סירופ", "גלזורה", "תערובת",
    "מכינים", "אופציות", "לגירסא", "לגרסה", "הרכבה", "ההרכבה", "קצפת", "פירורים",
]

# כותרות שמה שאחריהן הוא רשימת רכיבים או הוראות הכנה, גם כשהכול כתוב באותה שורה
INGREDIENT_HEADERS = ["החומרים", "חומרים", "רכיבים", "המרכיבים", "מרכיבים", "מצרכים", "המצרכים"]
INSTRUCTION_HEADERS = ["אופן ההכנה", "אופן הכנה", "ההכנה", "הכנה", "הוראות הכנה", "הוראות"]

# מילים שמרמזות ששורה היא רכיב (כמות או יחידת מידה)
QUANTITY_WORDS = [
    "כוס", "כוסות", "כף", "כפות", "כפית", "כפיות", "גרם", "ק\"ג", "קילו", "ליטר",
    "חבילה", "חבילות", "שקית", "שקיות", "קופסה", "קופסת", "קופסאות", "מעט", "קורט",
    "חצי", "רבע", "שליש", "יחידה", "יחידות", "פחית", "צנצנת", "שן", "שיני", "ראש",
    "צרור", "קמצוץ", "מיכל", "גביע", "גביעים", "בקבוק",
]

# מילים שמופיעות בשורות קצרות של מתכון, ולכן שורה שמכילה אותן אינה שם של מחברת
NOT_AUTHOR_WORDS = ["תבנית", "חבילת", "כמות", "לחם", "גבינה", "גבינת", "כרוב", "בשר", "עוף",
                    "שניצל", "כבד", "המבורגר", "שומשום", "סלסלת", "קוביות", "טעים", "מעלי", "עלי",
                    "ללא", "בלי", "וקל", "קל", "מנה", "מנות", "שרבט", "חום-", "תפו\"א"]


# ---------- קריאת המסמכים ----------

def read_docx_lines(path):
    """מחזירה רשימת שורות לפי סדר המסמך, כולל טבלאות. לכל שורה: טקסט, מודגשת, מסומנת בקו תחתון."""
    document = docx.Document(path)
    lines = []

    def add_paragraph(paragraph):
        runs = [r for r in paragraph.runs if r.text.strip()]
        is_bold = bool(runs) and all(r.bold for r in runs)
        is_underline = bool(runs) and any(r.underline for r in runs)
        # פסקה אחת במסמך יכולה להכיל ירידות שורה פנימיות
        for text in paragraph.text.split("\n"):
            lines.append({"text": text.replace("\xa0", " ").strip(),
                          "bold": is_bold, "underline": is_underline})

    # מעבר על גוף המסמך לפי הסדר, כדי שמתכונים בתוך טבלאות יופיעו במקומם
    for element in document.element.body.iterchildren():
        tag = element.tag.split("}")[-1]
        if tag == "p":
            add_paragraph(Paragraph(element, document))
        elif tag == "tbl":
            table = Table(element, document)
            seen_cells = set()
            for row in table.rows:
                for cell in row.cells:
                    if id(cell._tc) in seen_cells:  # תאים ממוזגים חוזרים כמה פעמים
                        continue
                    seen_cells.add(id(cell._tc))
                    for paragraph in cell.paragraphs:
                        add_paragraph(paragraph)
            lines.append({"text": "", "bold": False, "underline": False})
    return lines


def remove_notes(lines):
    """מסירה הערות עבודה שאינן חלק ממתכון, כמו 'מכאן עדיין לא הודפס!!!',
    גם כשהן מפוצלות על כמה שורות ('מכאן' / 'לא' / 'הודפס' / '!!!!')."""
    result, i = [], 0
    while i < len(lines):
        text = lines[i]["text"]
        if re.search(r"מכאן.*הודפס", text):
            i += 1
            continue
        window = " ".join(l["text"] for l in lines[i:i + 4])
        if text == "מכאן" and re.match(r"^מכאן\s+(עדיין\s+)?לא\s+הודפס", window):
            i += 1
            while i < len(lines) and re.fullmatch(r"(עדיין|לא|הודפס|!+)?", lines[i]["text"]):
                if not lines[i]["text"]:  # שורה ריקה מסיימת את ההערה
                    break
                i += 1
            continue
        result.append(lines[i])
        i += 1
    return result


def read_txt_lines(path):
    """בקובצי TXT אין הדגשות, ולכן כותרת מזוהה לפי שורה קצרה אחרי שורה ריקה (ראו is_title_line)."""
    text = path.read_text(encoding="utf-8-sig")
    return [{"text": t.strip(), "bold": False, "underline": False} for t in text.splitlines()]


# ---------- סיווג שורות ----------

def is_section_header(text):
    """כותרת של חלק בתוך מתכון, כמו 'מילוי:' או 'לבצק:'."""
    clean = text.strip().rstrip(":").strip()
    if len(clean.split()) > 6:
        return False
    for word in SECTION_WORDS:
        if clean.startswith(word) or clean.startswith("ל" + word) or clean.startswith("ה" + word):
            return True
    return False


def looks_like_ingredient(text):
    """שורה קצרה שמתחילה במספר או ביחידת מידה, או שורה קצרה מאוד בלי פעולה."""
    words = text.split()
    if not words or text.startswith("•"):
        return False
    if re.match(r"^[\d½¼¾⅓⅔⅛]", text) and len(words) <= 14:
        return True
    if words[0] in QUANTITY_WORDS and len(words) <= 8:
        return True
    # שורה של מילה או שתיים ("מלח", "פלפל שחור") היא כמעט תמיד רכיב
    return len(words) <= 2 and not text.endswith(":")


def is_title_line(line, prev_blank_count, next_text, is_txt):
    text = line["text"]
    if not text or len(text) > 90:
        return False
    # 'מילוי:' או 'בצק' הם חלק במתכון, אבל 'בצק פילו במילוי מתוק' הוא שם של מתכון
    if is_section_header(text) and (text.endswith(":") or len(text.split()) <= 2):
        return False
    if line["bold"]:
        # שורה מודגשת שמתחילה בכמות ('1/2 ק"ג בצק שטרודל') היא רכיב ולא כותרת
        return not re.match(r"^[\d½¼¾⅓⅔⅛]", text)
    if text.endswith(":") or looks_like_ingredient(text) or is_section_header(text):
        return False
    # כותרת לא מודגשת: שורה קצרה אחרי כמה שורות ריקות
    if prev_blank_count >= 3 and len(text.split()) <= 10:
        return True
    # כותרת לא מודגשת אחרי שורה ריקה אחת, כשמיד אחריה מתחילה רשימת רכיבים עם כמויות
    if prev_blank_count >= 1 and len(text.split()) <= 6 and re.match(r"^[\d½¼¾⅓⅔⅛]", next_text or "") \
            and not re.search(r"[\d:.#]", text) and not is_instruction_verb(text.split()[0]) \
            and text.split()[0] not in ("או", "מיץ", "קופסא", "קופסת", "חבילת", "עוד", "אחכ", "אח\"כ", "ואז", "רסק"):
        return True
    # כותרת לא מודגשת שמיד אחריה מתחיל חלק של מתכון ('שכבה א':', 'פירורים:')
    words = text.split()
    next_is_part = ":" in (next_text or "") and is_section_header(next_text.split(":")[0])
    starts_like_name = (not re.match(r"^[\d½¼¾⅓⅔⅛(]", text) and ":" not in text and not is_instruction_verb(words[0])
                        and not re.fullmatch(r"מ[א-ת]{3,}ים", words[0])  # 'מבשלים', 'מערבבים' פותחים הוראה
                        and not set(words) & {"לאפות", "דקות", "מעלות", "מתאים", "מתאימה"})
    if prev_blank_count >= 1 and len(words) <= 8 and next_is_part and starts_like_name:
        return True
    # 'מוס שוקולד- ענבל': שם ומחברת אחרי מקף, גם בלי שורה ריקה לפני, כשאחריו מתחילים רכיבים או חלק
    dash = re.match(r"^([^\d]+?)-\s*([א-ת'\"]+(?: [א-ת'\"]+){0,2})$", text)
    if dash and starts_like_name and len(words) <= 7 and not is_instruction_verb(dash.group(2).split()[0]) \
            and (next_is_part or re.match(r"^[\d½¼¾⅓⅔⅛]", next_text or "")) \
            and not any(w in QUANTITY_WORDS or w in NOT_AUTHOR_WORDS for w in dash.group(2).split()):
        return True
    # בקובץ TXT: שורה קצרה אחרי שורה ריקה, שאינה נראית כמו רכיב
    if is_txt and prev_blank_count >= 1 and not looks_like_ingredient(text) and len(text.split()) <= 8:
        return True
    return False


def is_instruction_verb(word):
    """מילה כמו 'לטגן', 'להשרות' או 'לערבב' פותחת הוראה, ולא שם של מתכון או של מחברת."""
    return bool(re.fullmatch(r"ל(ה|ט|ח|ק|ע|ב|ש|פ|ג|ר|מ|צ|ס|ז|נ)\S{2,}", word)) and word not in ("לאה", "לחם", "לחמניות")


def looks_like_author(text):
    """שם מחברת: 2–3 מילים בעברית, בלי מספרים ובלי סימני פיסוק של הוראה."""
    if re.search(r"[\d:;,.!?•]", text):
        return False
    words = text.split()
    if not 2 <= len(words) <= 3:
        return False
    if any(w in QUANTITY_WORDS or w in NOT_AUTHOR_WORDS for w in words):
        return False
    if is_instruction_verb(words[0]):
        return False
    return all(re.fullmatch(r"[א-ת'\"״׳\-]+", w) for w in words)


def split_title_and_author(title):
    """כותרות כמו 'עוגת גבינה        סבתא פרידלנדר', שבהן המחברת מופרדת ברווח גדול."""
    parts = re.split(r"\s{4,}", title)
    if len(parts) >= 2:
        return parts[0].strip(), " ".join(p.strip() for p in parts[1:]).strip(" ()")
    # 'אייס קפה (אסתי שיינברג)'
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", title)
    if match and looks_like_author(match.group(2)):
        return match.group(1).strip(), match.group(2).strip()
    return title, None


# ---------- חילוץ מתכון בודד ----------

def split_inline_headers(text):
    """'החומרים: 4 עגבניות ... ההכנה: טוחנים...' -> [('ingredients', '4 עגבניות ...'), ('instructions', 'טוחנים...')]"""
    names = "|".join(INGREDIENT_HEADERS + INSTRUCTION_HEADERS)
    pieces = re.split(rf"(?:^|\s)({names})\s*:", text)
    if len(pieces) == 1:
        return [(None, text)]
    result = []
    if pieces[0].strip():
        result.append((None, pieces[0].strip()))
    for header, content in zip(pieces[1::2], pieces[2::2]):
        kind = "ingredients" if header in INGREDIENT_HEADERS else "instructions"
        if content.strip():
            result.append((kind, content.strip()))
    return result


def parse_recipe_body(lines):
    """מחלקת את גוף המתכון לחלקים. כל חלק: שם, רכיבים, הוראות."""
    parts = [{"name": None, "ingredients": [], "instructions": []}]
    for line in lines:
        text = line["text"]
        if not text:
            continue
        # כותרת חלק מסומנת בקו תחתון, לפעמים עם תוכן אחרי הנקודתיים ('בצק: 5 כפות סוכר')
        if line["underline"] and ":" in text and len(text.split(":")[0].split()) <= 6:
            name, rest = text.split(":", 1)
            parts.append({"name": name.strip(), "ingredients": [], "instructions": []})
            text = rest.strip()
            if not text:
                continue
        # כותרת חלק: קצרה, ומסומנת בקו תחתון או מסתיימת בנקודתיים
        elif len(text.split()) <= 6 and (line["underline"] or (text.endswith(":") and is_section_header(text))):
            parts.append({"name": text.rstrip(":").strip(), "ingredients": [], "instructions": []})
            continue
        # 'לרוטב: 1/4 כוס קטשופ' — כותרת חלק ורכיב באותה שורה
        match = re.match(r"^([^:]{2,25}):\s*(.+)$", text)
        if match and is_section_header(match.group(1)) and match.group(1).strip() not in INGREDIENT_HEADERS + INSTRUCTION_HEADERS:
            parts.append({"name": match.group(1).strip(), "ingredients": [], "instructions": []})
            text = match.group(2).strip()
        for kind, piece in split_inline_headers(text):
            if kind is None:
                kind = "ingredients" if looks_like_ingredient(piece) else "instructions"
            parts[-1][kind].append(piece)
    return [p for p in parts if p["name"] or p["ingredients"] or p["instructions"]]


def build_recipe(title_line, body_lines, source_file, category, occurrence):
    title, author = split_title_and_author(title_line["text"].rstrip(":").strip())
    body = list(body_lines)

    # מחברת בשורה שמתחת לכותרת
    first = next((i for i, l in enumerate(body) if l["text"]), None)
    if author is None and first is not None and looks_like_author(body[first]["text"]):
        author = body[first]["text"].strip(" ()")
        body.pop(first)

    parts = parse_recipe_body(body)
    ingredients = [i for p in parts for i in p["ingredients"]]
    instructions = [i for p in parts for i in p["instructions"]]
    named_parts = [p for p in parts if p["name"]]

    original_text = "\n".join([title_line["text"]] + [l["text"] for l in body_lines]).strip()
    original_text = re.sub(r"\n{3,}", "\n\n", original_text)

    # מזהה יציב: אותו קובץ + אותה כותרת + מספר ההופעה של הכותרת בקובץ
    key = f"{source_file}|{title}|{occurrence}"
    recipe_id = "r_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

    # None = השדה לא נמצא במסמך. שגיאת חילוץ נרשמת בנפרד ב־extraction_status
    return {
        "recipe_id": recipe_id,
        "title": title,
        "source_file": source_file,
        "category": category,
        "author": author,
        "ingredients": ingredients or None,
        "instructions": instructions or None,
        "parts": named_parts or None,
        "original_text": original_text,
    }


def check_recipe(recipe, title_was_bold, line_count):
    """מחזירה (סיבות לחשד, הערות). הערה = שדה שלא קיים במסמך, וזה לא בהכרח בעיה."""
    reasons, notes = [], []
    if not recipe["ingredients"] and not recipe["instructions"]:
        reasons.append("אין תוכן מתחת לכותרת")
    elif not recipe["ingredients"]:
        notes.append("לא נמצאה רשימת רכיבים נפרדת (ייתכן שהרכיבים כתובים בתוך ההוראות)")
    elif not recipe["instructions"]:
        notes.append("לא נמצאו הוראות הכנה")
    if line_count > 70:
        reasons.append(f"מתכון ארוך מאוד ({line_count} שורות), ייתכן שאוחדו כמה מתכונים")
    if not title_was_bold:
        reasons.append("הכותרת לא מודגשת")
    return reasons, notes


# ---------- חילוץ קובץ שלם ----------

def extract_file(path):
    is_txt = path.suffix.lower() == ".txt"
    lines = remove_notes(read_txt_lines(path) if is_txt else read_docx_lines(path))
    category = path.stem

    # חלוקה לבלוקים לפי שורות כותרת
    blocks, preamble = [], []
    blank_count = 0
    for index, line in enumerate(lines):
        next_text = next((l["text"] for l in lines[index + 1:] if l["text"]), "")
        if is_title_line(line, blank_count, next_text, is_txt):
            blocks.append({"title": line, "body": []})
        elif blocks:
            blocks[-1]["body"].append(line)
        elif line["text"]:
            preamble.append(line["text"])
        blank_count = blank_count + 1 if not line["text"] else 0

    recipes, problems = [], []
    if preamble:
        problems.append({"title": "(טקסט לפני המתכון הראשון)", "status": "חשוד",
                         "reasons": ["טקסט שלא שויך לאף מתכון"], "preview": " / ".join(preamble[:3])})

    title_counts = {}
    for block in blocks:
        title_text = block["title"]["text"]
        # מספר ההופעה נספר לפי שם המתכון, כדי שלשני מתכונים באותו שם יהיו מזהים שונים
        clean_title = split_title_and_author(title_text.rstrip(":").strip())[0]
        title_counts[clean_title] = title_counts.get(clean_title, 0) + 1
        line_count = sum(1 for l in block["body"] if l["text"])
        try:
            recipe = build_recipe(block["title"], block["body"], path.name, category, title_counts[clean_title])
            reasons, notes = check_recipe(recipe, block["title"]["bold"] or is_txt, line_count)
            if not recipe["ingredients"] and not recipe["instructions"]:
                status = "נכשל"
            elif reasons:
                status = "חשוד"
            else:
                status = "תקין"
        except Exception as error:  # שגיאה בחילוץ, להבדיל משדה שפשוט לא קיים במסמך
            recipe = None
            status, reasons, notes = "נכשל", [f"שגיאה בחילוץ: {error}"], []

        if recipe:
            recipe["extraction_status"] = status
            recipe["extraction_notes"] = reasons + notes
            recipes.append(recipe)
        if status != "תקין":
            preview = recipe["original_text"][:150].replace("\n", " / ") if recipe else ""
            problems.append({"title": title_text, "status": status, "reasons": reasons, "preview": preview})
    return recipes, problems


def write_report(all_recipes, problems_by_file):
    by_status = {}
    for r in all_recipes:
        by_status[r["extraction_status"]] = by_status.get(r["extraction_status"], 0) + 1

    lines = ["דוח חילוץ מתכונים", "=" * 40, "", f"סה\"כ מתכונים: {len(all_recipes)}"]
    lines += [f"  {status}: {count}" for status, count in by_status.items()]
    no_ingredients = sum(1 for r in all_recipes if r["extraction_status"] != "נכשל" and not r["ingredients"])
    no_instructions = sum(1 for r in all_recipes if r["extraction_status"] != "נכשל" and not r["instructions"])
    lines.append(f"  מתוכם בלי רשימת רכיבים נפרדת: {no_ingredients}, בלי הוראות הכנה: {no_instructions}")
    lines += ["", "מתכונים במצב 'נכשל' לא ייכנסו לאינדקס. מתכונים 'חשודים' ייכנסו, אבל כדאי לבדוק אותם.", ""]

    for file_name, (count, problems) in problems_by_file.items():
        lines.append(f"--- {file_name}: {count} מתכונים, {len(problems)} לבדיקה ---")
        for p in problems:
            lines.append(f"[{p['status']}] {p['title']}")
            lines.append(f"    סיבה: {'; '.join(p['reasons'])}")
            if p["preview"]:
                lines.append(f"    תחילת הטקסט: {p['preview']}")
        lines.append("")
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def extract_all(force=False):
    if RECIPES_FILE.exists() and not force:
        print(f"הקובץ {RECIPES_FILE.name} כבר קיים, ולכן לא חולץ מחדש כדי לא לדרוס תיקונים ידניים.")
        print("לחילוץ מחדש הריצי: python extract_recipes.py --force")
        return

    files = sorted(list(DATA_DIR.glob("*.docx")) + list(DATA_DIR.glob("*.txt")))
    files = [f for f in files if not f.name.startswith("~$")]  # קבצים זמניים של Word פתוח
    if not files:
        print(f"לא נמצאו קובצי DOCX או TXT בתיקייה {DATA_DIR}")
        return

    all_recipes, problems_by_file = [], {}
    for path in files:
        try:
            recipes, problems = extract_file(path)
        except Exception as error:
            print(f"שגיאה בקריאת הקובץ {path.name}: {error}")
            problems_by_file[path.name] = (0, [{"title": "(כל הקובץ)", "status": "נכשל",
                                               "reasons": [str(error)], "preview": ""}])
            continue
        all_recipes.extend(recipes)
        problems_by_file[path.name] = (len(recipes), problems)
        print(f"{path.name}: {len(recipes)} מתכונים, {len(problems)} לבדיקה")

    RECIPES_FILE.write_text(json.dumps(all_recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(all_recipes, problems_by_file)
    print(f"\nנשמרו {len(all_recipes)} מתכונים בקובץ {RECIPES_FILE}")
    print(f"דוח החילוץ נשמר בקובץ {REPORT_FILE}")


if __name__ == "__main__":
    # בטרמינל של Windows, הדפסה בעברית דורשת קידוד UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    extract_all(force="--force" in sys.argv)
