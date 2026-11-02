"""
בדיקות אחזור על שאלות מוגדרות מראש. מדפיס את התוצאות ושומר דוח CSV לסקירה ידנית.
הרצה: python evaluate.py
את איכות התשובות של Gemini בודקים ידנית בממשק Streamlit.
"""

import csv
import sys
from pathlib import Path

import rag
from index_recipes import describe_api_error

REPORT_FILE = Path(__file__).parent / "evaluation_report.csv"

# כל שאלה: סוג הבדיקה, ומה מצפים. excluded / required נבדקים מול המתכון המלא בכל תוצאה.
# expect_empty = שאלה שהתשובה עליה לא נמצאת במאגר.
TEST_QUESTIONS = [
    {"type": "חיפוש ישיר", "question": "עוגת גבינה"},
    {"type": "חיפוש ישיר", "question": "איך מכינים חלות לשבת?"},
    {"type": "ניסוח שונה לאותו רעיון", "question": "אני רוצה לאפות חלה"},
    {"type": "חיפוש לפי רכיב", "question": "איזה מתכונים יש עם תפוחים?", "required": ["תפוחים"]},
    {"type": "חיפוש לפי רכיב", "question": "איזה קינוח קר אפשר להכין עם שמנת ושוקולד?",
     "required": ["שמנת", "שוקולד"]},
    {"type": "רכיב אסור", "question": "אני מחפשת עוגת שוקולד בלי מרגרינה", "excluded": ["מרגרינה"]},
    {"type": "רכיב אסור", "question": "יש לך מתכון ללא קמח?", "excluded": ["קמח"]},
    {"type": "רכיב אסור", "question": "עוגיות בלי אגוזים", "excluded": ["אגוזים"]},
    {"type": "חיפוש לפי קטגוריה", "question": "תציע לי כמה סלטים שמתאימים לאירוח"},
    {"type": "קטגוריה מפורשת (Metadata Filtering)", "question": "משהו עם שוקולד מהקטגוריה פטיפורים",
     "category": "פטיפורים"},
    {"type": "העדפה", "question": "אני רוצה משהו מתוק וקל להכנה"},
    {"type": "אין במאגר", "question": "יש לך מתכון לסושי?", "expect_empty": True},
    {"type": "אין במאגר", "question": "איך מכינים לזניה בלי גבינה ובלי בצק?", "expect_empty": True},
]

# שאלות המשך נבדקות כשיחה: השאלה השנייה מתייחסת לתוצאות של הראשונה
FOLLOW_UP_CONVERSATIONS = [
    ["תציעי לי עוגות שוקולד", "תציגי את השני במלואו"],
    ["תציעי לי עוגות גבינה", "מה ההבדל בין שתי הראשונות?"],
]


def evaluate_question(index, recipes, test):
    """מריצה שאלה אחת דרך זיהוי הדרישות והאחזור (בלי יצירת תשובה), ובודקת את התוצאות מול הציפיות."""
    categories = sorted({r["category"] for r in recipes.values()})
    requirements = rag.parse_requirements(test["question"], categories)
    results = rag.find_recipes(index, recipes, requirements)

    expected = rag.Requirements(search_text="", excluded_ingredients=test.get("excluded", []),
                                required_ingredients=test.get("required", []))
    problems = []
    for item in results:
        check = rag.check_ingredients(item["recipe"], expected)
        if not check["ok"]:
            problems.append(f"{item['recipe']['title']}: {check}")
        if test.get("category") and item["recipe"]["category"] != test["category"]:
            problems.append(f"{item['recipe']['title']}: קטגוריה {item['recipe']['category']}")
    if test.get("expect_empty") and results:
        problems.append("ציפינו שלא יימצאו מתכונים, אבל נמצאו (לבדוק ידנית אם הם באמת רלוונטיים)")

    return {
        "סוג": test["type"],
        "שאלה": test["question"],
        "דרישות שזוהו": f"אסור: {requirements.excluded_ingredients}, חובה: {requirements.required_ingredients}, "
                        f"קטגוריה: {requirements.source_category}",
        "תוצאות לפי הסדר": " | ".join(f"{i['score']:.3f} {i['recipe']['title']}" for i in results),
        "מקורות": " | ".join(rag.source_label(i["recipe"], recipes) for i in results),
        "עומד בדרישות": "כן" if not problems else "לא",
        "הערות": "; ".join(problems),
    }


def evaluate_conversation(index, recipes, questions):
    """מריצה שיחה קצרה עם answer_question, כדי לבדוק שאלות המשך: על אילו מתכונים התבססה כל תשובה."""
    rows, history, shown_ids = [], [], []
    for question in questions:
        result = rag.answer_question(index, recipes, question, history, shown_ids)
        rows.append({
            "סוג": "שאלת המשך" if history else "פתיחת שיחה",
            "שאלה": question,
            "דרישות שזוהו": "",
            "תוצאות לפי הסדר": result["answer"][:200].replace("\n", " "),
            "מקורות": " | ".join(s["label"] for s in result["sources"]),
            "עומד בדרישות": "לבדיקה ידנית",
            "הערות": "",
        })
        history += [("משתמשת", question), ("מערכת", result["answer"])]
        shown_ids = result["recipe_ids"] or shown_ids
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    index = rag.load_index()
    rag.setup_llm()
    recipes = rag.load_recipes()

    rows = []
    for test in TEST_QUESTIONS:
        try:
            row = evaluate_question(index, recipes, test)
        except Exception as error:  # שגיאה בשאלה אחת לא עוצרת את כל הבדיקה
            row = {"סוג": test["type"], "שאלה": test["question"], "הערות": f"שגיאה: {describe_api_error(error)}"}
        rows.append(row)
        print(f"[{row.get('עומד בדרישות', 'שגיאה')}] {row['שאלה']}\n    {row.get('תוצאות לפי הסדר', '')}\n")

    for conversation in FOLLOW_UP_CONVERSATIONS:
        try:
            rows.extend(evaluate_conversation(index, recipes, conversation))
        except Exception as error:
            rows.append({"סוג": "שאלת המשך", "שאלה": conversation[0], "הערות": f"שגיאה: {describe_api_error(error)}"})

    # utf-8-sig כדי שהעברית תוצג נכון כשפותחים את הקובץ ב־Excel
    with open(REPORT_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["סוג", "שאלה", "דרישות שזוהו", "תוצאות לפי הסדר",
                                                  "מקורות", "עומד בדרישות", "הערות"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"הדוח נשמר בקובץ {REPORT_FILE}")
