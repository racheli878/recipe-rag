"""
טעינת האינדקס הקיים, אחזור מתכונים ויצירת תשובות באמצעות Gemini ו־LlamaIndex.
"""

import json
import os
import re
from typing import List, Optional

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from pydantic import BaseModel, Field

from index_recipes import RECIPES_FILE, describe_api_error, get_vector_store, setup_embedding


def load_index():
    """טוענת את האינדקס הקיים מ־ChromaDB, בלי לחשב Embeddings מחדש.
    מחזירה VectorStoreIndex של LlamaIndex. אם האינדקס עוד לא נבנה, מעלה שגיאה עם הסבר."""
    setup_embedding()  # אותו מודל Embeddings כמו באינדוקס, כדי שהשאילתות יושוו לאותו מרחב וקטורי
    _, collection, vector_store = get_vector_store()
    if collection.count() == 0:
        raise RuntimeError("האינדקס עדיין לא נבנה. הריצי קודם: python index_recipes.py")
    return VectorStoreIndex.from_vector_store(vector_store)


def load_recipes():
    """טוענת את המתכונים המלאים מ־recipes.json למילון לפי recipe_id.
    המתכון המלא משמש לבדיקת רכיבים ולהצגת מתכון, ולא רק ה־Chunk שנשלף."""
    if not RECIPES_FILE.exists():
        raise RuntimeError("לא נמצא data/recipes.json. הריצי קודם: python extract_recipes.py")
    recipes = json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    return {r["recipe_id"]: r for r in recipes}


def setup_llm():
    """מגדירה את Gemini כמודל השפה של LlamaIndex (Settings.llm). המפתח כבר נטען ב־setup_embedding."""
    from llama_index.llms.google_genai import GoogleGenAI
    # בלי temperature: במודלי Gemini 3 ההמלצה היא להשאיר את ברירת המחדל, וערך נמוך עלול לגרום לתקיעה.
    # max_retries=1: בלי ניסיונות חוזרים. timeout של 30 שניות (באלפיות שנייה), כדי שבקשה תקועה תיכשל עם הודעה
    Settings.llm = GoogleGenAI(model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), max_retries=1,
                               http_options={"timeout": 30_000})


class Requirements(BaseModel):
    """הדרישות שזוהו בשאלה. LlamaIndex מבקשת מ־Gemini למלא בדיוק את המבנה הזה (פלט מובנה)."""
    search_text: str = Field(description="מה מחפשים, בלי הרכיבים האסורים. למשל: 'עוגת שוקולד'")
    required_ingredients: List[str] = Field(default_factory=list, description="רכיבים שחייבים להופיע במתכון")
    excluded_ingredients: List[str] = Field(default_factory=list, description="רכיבים שאסור שיופיעו במתכון")
    preferences: List[str] = Field(default_factory=list, description="העדפות שאינן תנאי מחייב, כמו 'קל להכנה'")
    source_category: Optional[str] = Field(default=None, description="קטגוריה מהרשימה, רק אם המשתמשת ביקשה אותה במפורש")
    is_follow_up: bool = Field(default=False, description="האם השאלה מתייחסת למתכונים שכבר הוצגו")
    recipe_number: Optional[int] = Field(default=None, description="מספר המתכון מהרשימה שהוצגה, אם השאלה על מתכון מסוים")
    wants_full_recipe: bool = Field(default=False, description="האם המשתמשת מבקשת להציג מתכון במלואו")
    wants_more: bool = Field(default=False, description="האם המשתמשת מבקשת עוד רעיונות לאותה בקשה")


REQUIREMENTS_PROMPT = PromptTemplate(
    "נתחי את השאלה הבאה של משתמשת שמחפשת מתכונים.\n"
    "- search_text: נושא החיפוש בלבד, בלי רכיבים אסורים ובלי מילים כמו 'בלי' או 'ללא'.\n"
    "- required_ingredients: רק רכיבים שהמשתמשת דורשת במפורש שיהיו במתכון.\n"
    "- excluded_ingredients: רכיבים שהמשתמשת אוסרת ('בלי', 'ללא', 'אין לי', 'בלי שום').\n"
    "- preferences: העדפות כלליות שאינן רכיב מחייב. למשל 'קר', 'קל', 'פשוט', 'לאירוח', 'הרבה יחידות' הן העדפות ולא רכיבים.\n"
    "- source_category: אחת מהקטגוריות האלה רק אם השאלה מבקשת אותה במפורש, אחרת null: {categories}\n"
    "- is_follow_up: true אם השאלה מתייחסת למתכונים שכבר הוצגו ('השני', 'במתכון הזה', 'שהצעת').\n"
    "- recipe_number: מספר המתכון ברשימה שהוצגה, אם השאלה על מתכון מסוים ('השני' = 2).\n"
    "- wants_full_recipe: true אם מבקשים להציג מתכון במלואו.\n"
    "- wants_more: true אם מבקשים עוד רעיונות או מתכונים אחרים לאותה בקשה ('תן עוד', 'יש עוד?', 'משהו אחר'). "
    "במקרה כזה is_follow_up הוא false, ו־search_text, הרכיבים והקטגוריה נלקחים מהשאלה הקודמת.\n"
    "כתבי שמות רכיבים בצורה בסיסית (למשל 'אגוזים', 'מרגרינה'). אל תמציאי דרישות שלא נאמרו.\n\n"
    "השאלה הקודמת: {previous}\n"
    "המתכונים שהוצגו לאחרונה: {shown}\n"
    "השאלה: {question}"
)


def parse_requirements(question, categories, shown_titles=(), previous=""):
    """מזהה בשאלה רכיבים רצויים ואסורים, העדפות וקטגוריה, באמצעות Gemini דרך structured_predict של LlamaIndex.
    אם הקריאה נכשלת, משתמשת בכללים פשוטים כגיבוי, כדי שהחיפוש ימשיך לעבוד."""
    try:
        requirements = Settings.llm.structured_predict(
            Requirements, REQUIREMENTS_PROMPT, question=question, categories=", ".join(categories),
            shown=" | ".join(f"{i}. {t}" for i, t in enumerate(shown_titles, 1)) or "אין", previous=previous or "אין")
    except Exception:
        return parse_requirements_with_rules(question)

    # בדיקות בקוד אחרי Gemini, כי המודל לא תמיד מקפיד על ההוראות:
    # רכיב חובה נשאר רק אם הוא באמת מופיע בשאלה (או בשאלה הקודמת, בבקשת "עוד רעיונות")
    asked = question + " " + (previous if requirements.wants_more else "")
    requirements.required_ingredients = [i for i in requirements.required_ingredients if mentions(asked, i)]
    # סינון קשיח לפי קובץ רק כשהמשתמשת ביקשה קטגוריה או קובץ במפורש
    if not re.search(r"קטגורי|קובץ", asked):
        requirements.source_category = None
    return requirements


def parse_requirements_with_rules(question):
    """גיבוי פשוט: 'בלי X' / 'ללא X' הם רכיבים אסורים, 'עם X' הוא רכיב רצוי."""
    excluded = re.findall(r"(?:בלי|ללא)\s+([א-ת\"']+)", question)
    required = re.findall(r"(?:^|\s)עם\s+([א-ת\"']+)", question)
    search_text = re.sub(r"(?:בלי|ללא)\s+[א-ת\"']+", "", question).strip()
    return Requirements(search_text=search_text or question,
                        required_ingredients=required, excluded_ingredients=excluded)


# כמה שמות קרובים לרכיבים נפוצים. זו לא רשימת תחליפים, רק צורות שונות של אותו רכיב
RELATED_NAMES = {
    "אגוזים": ["אגוזי מלך", "פקאן", "לוז", "קשיו", "פיסטוק", "בוטנים"],
    "ביצים": ["חלמון", "חלמונים", "חלבון", "חלבונים"],
    "חלב": ["חלבי"],
    "קמח": ["בצק", "פירורי לחם", "מצה"],  # 'ללא קמח' פוסל גם מתכון שמשתמש בבצק מוכן
    # אף מתכון לא כותב "גלוטן", אלא את הרכיבים שמכילים אותו
    "גלוטן": ["קמח", "פירורי לחם", "לחם", "סולת", "פתיתים", "פסטה", "אטריות", "בצק", "מצה", "שיבולת שועל"],
}

# צירופים שמכילים את שם הרכיב אבל הם רכיב אחר ('תפוחי אדמה' אינם תפוחים)
DIFFERENT_INGREDIENTS = {
    "תפוחים": ["תפוחי אדמה", "תפו\"א", "תפוא"],
}


def word_variants(word):
    """צורות פשוטות של מילה: יחיד/רבים וסמיכות ('אגוז', 'אגוזים', 'אגוזי')."""
    stem = re.sub(r"(ים|ות|י|ה)$", "", word)  # 'גבינה' -> 'גבינ' -> 'גבינות'
    return {word, stem, stem + "ים", stem + "ות", stem + "י", stem + "ה", stem + "ת"}


def mentions(text, ingredient):
    """האם הרכיב מופיע בטקסט, כמילה שלמה, גם עם תחיליות כמו ו/ב/ל/ה/מ/ש ('ושוקולד', 'בשמנת')."""
    for other in DIFFERENT_INGREDIENTS.get(ingredient, []):
        text = text.replace(other, " ")
    names = [ingredient] + RELATED_NAMES.get(ingredient, [])
    for name in names:
        words = name.split()
        # הצורות האפשריות של המילה האחרונה בשם (בשם של כמה מילים, רק היא משתנה בדרך כלל)
        endings = "|".join(re.escape(v) for v in word_variants(words[-1]) if len(v) >= 2)
        start = re.escape(" ".join(words[:-1]) + " ") if len(words) > 1 else ""
        pattern = rf"(?<![א-ת])[ובלהמשכ]{{0,2}}{start}(?:{endings})(?![א-ת])"
        if re.search(pattern, text):
            return True
    return False


def check_ingredients(recipe, requirements):
    """בודקת את דרישות הרכיבים מול המתכון המלא (רשימת הרכיבים והטקסט המקורי), ולא רק מול ה־Chunk שנשלף.
    מחזירה: ok (עומד בכל הדרישות), violations (רכיבים אסורים שנמצאו), missing (רכיבים רצויים שחסרים),
    ו־uncertain: כשלמתכון אין רשימת רכיבים נפרדת, אי אפשר לוודא בביטחון שרכיב אסור לא מופיע."""
    text = recipe["original_text"] + "\n" + "\n".join(recipe["ingredients"] or [])
    violations = [i for i in requirements.excluded_ingredients if mentions(text, i)]
    missing = [i for i in requirements.required_ingredients if not mentions(text, i)]
    uncertain = bool(requirements.excluded_ingredients) and not recipe["ingredients"]
    return {"ok": not violations and not missing, "violations": violations,
            "missing": missing, "uncertain": uncertain}


def search_recipes(index, recipes, question, top_k=10, filters=None):
    """חיפוש סמנטי: שולף Chunks עם ה־Retriever של LlamaIndex, ומאחד Chunks של אותו מתכון.
    filters (אופציונלי) הוא MetadataFilters, שמסנן כבר בתוך ChromaDB לפני הדירוג.
    מחזירה רשימת מתכונים שונים לפי סדר הדמיון, כל אחד עם הציון הגבוה מבין ה־Chunks שלו."""
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    results = retriever.retrieve(question)

    found = {}  # recipe_id -> תוצאה. האיחוד לפי מזהה ולא לפי שם, כי יש מתכונים שונים באותו שם
    for result in results:
        recipe_id = result.node.metadata["recipe_id"]
        if recipe_id not in found:
            found[recipe_id] = {"recipe": recipes[recipe_id], "score": result.score, "chunks": []}
        found[recipe_id]["chunks"].append(result.node.get_content())
        found[recipe_id]["score"] = max(found[recipe_id]["score"], result.score)

    return sorted(found.values(), key=lambda item: item["score"], reverse=True)


def find_recipes(index, recipes, requirements, max_results=4, progress=lambda step: None, exclude_ids=()):
    """חיפוש לפי הדרישות שזוהו, בדיקת רכיבים מול המתכון המלא, והחזרת עד 4 מתכונים מתאימים.
    אם אחרי הסינון נשארו מעט מתכונים, מחפשים שוב עם יותר מועמדים (חיפוש מדורג).
    exclude_ids: מתכונים שכבר הוצגו, כשהמשתמשת מבקשת עוד רעיונות."""
    categories = sorted({r["category"] for r in recipes.values()})

    # Metadata Filtering: רק כשהמשתמשת ביקשה קטגוריה במפורש, כדי לא לפספס מתכונים מקבצים אחרים
    filters = None
    if requirements.source_category in categories:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value=requirements.source_category)])

    for top_k in (10, 30):
        matching = []
        progress("מחפש מתכונים...")
        found = search_recipes(index, recipes, requirements.search_text, top_k, filters)
        progress("בודק התאמה לדרישות...")
        for item in found:
            if item["recipe"]["recipe_id"] in exclude_ids:
                continue
            item["check"] = check_ingredients(item["recipe"], requirements)
            if item["check"]["ok"]:
                matching.append(item)
        if len(matching) >= max_results:
            break

    return matching[:max_results]


ANSWER_PROMPT = PromptTemplate(
    "את עוזרת מתכונים. עני בעברית, בסגנון שיחה טבעי וזורם וקצר, רק על סמך המתכונים שלמטה.\n"
    "כללים:\n"
    "- אל תמציאי מתכונים, רכיבים, כמויות, הוראות, זמני הכנה או מחברות שלא מופיעים במתכונים.\n"
    "- אם המידע לא מספיק כדי לענות, אמרי זאת במפורש.\n"
    "- אם אף מתכון לא עונה על השאלה, אמרי זאת במשפט אחד, בלי להציע מתכונים שאינם קשורים לשאלה.\n"
    "- אם השאלה היא על המאגר או על המערכת עצמה ולא בקשה למתכון (למשל 'כמה מתכונים יש לך?'), אמרי שאת יכולה "
    "לעזור למצוא מתכונים מהמאגר, ואל תסיקי מספרים או מסקנות מהמתכונים שלמטה.\n"
    "- כשממליצים: פתיח קצר ואז פסקה זורמת ולא רשימה ממוספרת (למשל: 'יש את X, וגם Y יכולה להיות מוצלחת כי...'). "
    "לכל מתכון משפט אחד קצר: שמו ולמה הוא מתאים. אם יש מחברת, הזכירי אותה. אם אין, אל תכתבי על כך דבר.\n"
    "- אל תפרטי רשימת רכיבים. הזכירי רכיב רק אם השאלה עוסקת בו.\n"
    "- אל תכתבי שמות קבצים, הם מופיעים ברשימת המקורות.\n"
    "- הבחיני בין עובדה שכתובה במתכון לבין הערכה שלך (למשל 'נראה מתאים לאירוח').\n"
    "- כשממליצים, הזכירי את כל המתכונים שלמטה, לפי הסדר שבו הם מופיעים, כדי ששאלות המשך כמו 'השני' יתייחסו למתכון הנכון.\n"
    "- אל תכתבי רשימת מקורות, היא תתווסף אוטומטית.\n\n"
    "המתכונים (ממוספרים):\n{context_str}\n\n"
    "{query_str}\n"
)


def recipes_to_nodes(recipe_list):
    """הופכת מתכונים מלאים ל־Nodes, כדי ש־Gemini יענה על סמך המתכון המלא ולא רק על Chunk בודד."""
    nodes = []
    for number, recipe in enumerate(recipe_list, 1):
        author = f"מחברת: {recipe['author']}\n" if recipe["author"] else ""
        text = f"מתכון {number}: {recipe['title']}\nקובץ: {recipe['source_file']}\n{author}\n{recipe['original_text']}"
        nodes.append(NodeWithScore(node=TextNode(text=text), score=1.0))
    return nodes


def generate_answer(question, recipe_list, history=(), requirements=None):
    """יוצרת תשובה עם Gemini באמצעות Response Synthesizer של LlamaIndex.
    history: ההודעות האחרונות בשיחה (עד 6), כדי ש־Gemini יבין שאלות המשך.
    requirements: דרישות הרכיבים שהקוד כבר בדק, כדי ש־Gemini לא יבדוק אותן שוב בעצמו ויטעה."""
    synthesizer = get_response_synthesizer(response_mode="compact", text_qa_template=ANSWER_PROMPT)
    recent = "\n".join(f"{role}: {text}" for role, text in list(history)[-6:])
    query = (f"השיחה עד עכשיו:\n{recent}\n\n" if recent else "") + f"השאלה: {question}"
    if requirements and (requirements.excluded_ingredients or requirements.required_ingredients):
        checked = []
        if requirements.excluded_ingredients:
            checked.append("אין בהם: " + ", ".join(requirements.excluded_ingredients))
        if requirements.required_ingredients:
            checked.append("יש בהם: " + ", ".join(requirements.required_ingredients))
        query += ("\n\nכל המתכונים כבר נבדקו מול הרכיבים המלאים, ו" + "; ".join(checked) + ". "
                  "אל תבדקי זאת שוב, ואל תניחי שרכיב אחר (למשל שמן או חמאה במקום מרגרינה) הוא אותו רכיב.")
    response = synthesizer.synthesize(query, nodes=recipes_to_nodes(recipe_list))
    return str(response)


def ltr(text):
    """עוטפת טקסט כמו 'עוגות.docx' בסימני כיווניות, כדי שלא יתהפך ל־'docx.עוגות' בתצוגה מימין לשמאל."""
    return f"\u2066{text}\u2069"


def format_full_recipe(recipe):
    """המתכון המלא מתוך recipes.json, בלי שינוי של רכיבים, כמויות או הוראות. בלי קריאה ל־Gemini."""
    author = f"מחברת: {recipe['author']}  \n" if recipe["author"] else ""
    body = recipe["original_text"].split("\n", 1)[1] if "\n" in recipe["original_text"] else ""
    return f"### {recipe['title']}\n{author}מקור: {ltr(recipe['source_file'])}\n\n" + body.strip().replace("\n", "  \n")


def source_label(recipe, recipes):
    """שם המקור להצגה. כשיש כמה מתכונים באותו שם, מוסיפים מאפיין מזהה: מחברת או תחילת המתכון."""
    label = f"{recipe['title']} ({ltr(recipe['source_file'])}"
    same_name = [r for r in recipes.values() if r["title"] == recipe["title"]]
    if recipe["author"]:
        label += f", {recipe['author']}"
    elif len(same_name) > 1:
        first_line = (recipe["ingredients"] or recipe["instructions"] or [""])[0]
        label += f", מתחיל ב: {first_line[:30]}"
    return label + ")"


def answer_question(index, recipes, question, history=(), shown_ids=(), progress=lambda step: None):
    """הפונקציה המרכזית של הצ'אט. מחזירה מילון: answer (טקסט), sources (רשימת מקורות, כל אחד עם
    label ו־recipe_id), recipe_ids (המתכונים שמוצגים עכשיו, לשאלות ההמשך הבאות).
    progress היא פונקציה שמקבלת את שם השלב הנוכחי, כדי שהממשק יציג שלבי עבודה אמיתיים."""
    progress("מבין את הבקשה...")
    shown = [recipes[i] for i in shown_ids if i in recipes]
    categories = sorted({r["category"] for r in recipes.values()})
    previous = next((text for role, text in reversed(list(history)) if role == "משתמשת"), "")
    requirements = parse_requirements(question, categories, [r["title"] for r in shown], previous)

    # שאלת המשך על מתכונים שכבר הוצגו: עונים על סמך המתכונים המלאים, בלי חיפוש חדש
    if requirements.is_follow_up and shown and not requirements.wants_more:
        number = requirements.recipe_number
        targets = [shown[number - 1]] if number and 1 <= number <= len(shown) else shown
        if requirements.wants_full_recipe and len(targets) == 1:
            answer = format_full_recipe(targets[0])
        else:
            progress("מכין תשובה...")
            answer = generate_answer(question, targets, history)
        return {"answer": answer,
                "sources": [{"label": source_label(r, recipes), "recipe_id": r["recipe_id"]} for r in targets],
                "recipe_ids": list(shown_ids), "requirements": requirements}

    # 'תן עוד רעיונות': חיפוש חדש לפי השאלה הקודמת, בלי המתכונים שכבר הוצגו
    exclude_ids = set(shown_ids) if requirements.wants_more else set()
    results = find_recipes(index, recipes, requirements, progress=progress, exclude_ids=exclude_ids)
    if not results:
        return {"answer": "לא מצאתי במאגר מתכונים שעומדים בכל הדרישות שלך.",
                "sources": [], "recipe_ids": [], "requirements": requirements}

    found = [item["recipe"] for item in results]
    progress("מכין תשובה...")
    answer = generate_answer(question, found, history, requirements)
    sources = []
    for item in results:
        note = " [לא ניתן לוודא את כל הרכיבים: אין במתכון רשימת רכיבים]" if item["check"]["uncertain"] else ""
        sources.append({"label": source_label(item["recipe"], recipes) + note, "recipe_id": item["recipe"]["recipe_id"]})
    return {"answer": answer, "sources": sources, "recipe_ids": [r["recipe_id"] for r in found],
            "requirements": requirements}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    index = load_index()
    setup_llm()
    recipes = load_recipes()

    # שיחה לדוגמה עם שאלות המשך. אפשר גם שאלות משלך: python rag.py "שאלה 1" "שאלה 2"
    questions = sys.argv[1:] or ["תציעי לי עוגות שוקולד בלי מרגרינה", "תציגי את השני במלואו",
                                 "האם יש בו אגוזים?", "יש לך מתכון לסושי?"]
    history, shown_ids = [], []
    for question in questions:
        try:
            result = answer_question(index, recipes, question, history, shown_ids)
        except Exception as error:
            print(f"שגיאה: {describe_api_error(error)}")
            break
        print(f"משתמשת: {question}\n")
        print(result["answer"])
        if result["sources"]:
            print("\nמקורות:")
            for source in result["sources"]:
                print(f"  - {source['label']}")
        print("\n" + "=" * 60 + "\n")
        history += [("משתמשת", question), ("מערכת", result["answer"])]
        shown_ids = result["recipe_ids"] or shown_ids
