"""
בניית האינדקס של המתכונים: Chunking, יצירת Embeddings ושמירה ב־ChromaDB, באמצעות LlamaIndex.
הקלט הוא data/recipes.json, שנוצר על ידי extract_recipes.py (ואולי תוקן ידנית).

    python index_recipes.py             בניית האינדקס (או המשך בנייה שנעצרה)
    python index_recipes.py --rebuild   מחיקת האינדקס ובנייה מחדש, למשל אחרי תיקון ידני של recipes.json
    python index_recipes.py --show      הצגת כמה Chunks לדוגמה מתוך האינדקס
    python index_recipes.py --limit 700 אינדקס חלקי של עד 700 Chunks, כדי להשאיר מכסה יומית לשאלות
    python index_recipes.py --update    עדכון רק של מתכונים שהשתנו (ושל מתכונים שהופרדו מהם) אחרי חילוץ מחדש
"""

import json
import sys
import time
from itertools import zip_longest
from pathlib import Path

from llama_index.core.schema import TextNode
from llama_index.core.utils import get_tokenizer

RECIPES_FILE = Path(__file__).parent / "data" / "recipes.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "recipes"


def setup_embedding():
    """מגדירה את מודל ה־Embeddings של Google ב־Settings של LlamaIndex.
    אותה פונקציה תשמש גם את rag.py, כדי שהאינדוקס והשאילתות ישתמשו באותו מודל."""
    import os
    import truststore
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    truststore.inject_into_ssl()  # שימוש בתעודות האבטחה של Windows (פתר את שגיאת ה־SSL)
    load_dotenv(Path(__file__).parent / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("חסר מפתח GOOGLE_API_KEY בקובץ .env")
    # retries=1: בלי ניסיונות חוזרים. בחריגה ממכסה השאלה נכשלת מיד עם הודעה ברורה,
    # ובבניית האינדקס יש המתנה וניסיון חוזר משלנו (ב־build_index)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"), retries=1,
                                                http_options={"timeout": 30_000})


def get_vector_store():
    """פותחת (או יוצרת) את אוסף המתכונים ב־ChromaDB המקומי, ועוטפת אותו ב־ChromaVectorStore של LlamaIndex."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection, ChromaVectorStore(chroma_collection=collection)

CHUNK_SIZE = 800      # נבחר אחרי בדיקה על המתכונים: ב־400 נחתכו גם מתכונים קצרים (בעברית כל מילה שווה הרבה טוקנים)
CHUNK_OVERLAP = 40


def split_into_sections(recipe):
    """מחלקת את הטקסט המקורי של המתכון לחלקים (בצק, מילוי, ציפוי...) לפי כותרות החלקים שזוהו בחילוץ.
    הטקסט שלפני הכותרת הראשונה הוא חלק בפני עצמו."""
    part_names = {p["name"] for p in (recipe["parts"] or [])}
    sections = [[]]
    for line in recipe["original_text"].split("\n"):
        header = line.split(":")[0].strip()
        if header in part_names and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(lines).strip() for lines in sections if "\n".join(lines).strip()]


def recipe_to_nodes(recipe, splitter):
    """הופכת מתכון אחד לרשימת TextNode של LlamaIndex.
    מתכון קצר נשאר Chunk אחד. מתכון ארוך מחולק לפי החלקים שלו, וחלקים קטנים סמוכים מאוחדים
    עד לגודל ה־Chunk. רק חלק שארוך בעצמו מדי נחתך על ידי SentenceSplitter בגבולות שורות."""
    count_tokens = lambda text: len(get_tokenizer()(text))

    if count_tokens(recipe["original_text"]) <= CHUNK_SIZE:
        pieces = [recipe["original_text"]]
    else:
        pieces = []
        for section in split_into_sections(recipe):
            if count_tokens(section) > CHUNK_SIZE:
                pieces.extend(splitter.split_text(section))
            elif pieces and count_tokens(pieces[-1] + "\n" + section) <= CHUNK_SIZE:
                pieces[-1] += "\n" + section
            else:
                pieces.append(section)

    metadata = {
        "recipe_id": recipe["recipe_id"],
        "recipe_title": recipe["title"],
        "source_file": recipe["source_file"],
        "category": recipe["category"],
        "author": recipe["author"] or "",  # ChromaDB לא מקבלת ערך None ב־Metadata
    }
    # שם המתכון והקטגוריה נכנסים לטקסט שעליו מחושב ה־Embedding. המזהים הטכניים לא
    hidden_keys = ["recipe_id", "source_file", "chunk_index"] + ([] if recipe["author"] else ["author"])

    nodes = []
    for index, text in enumerate(pieces):
        node = TextNode(text=text, metadata={**metadata, "chunk_index": index})
        node.excluded_embed_metadata_keys = hidden_keys
        node.excluded_llm_metadata_keys = hidden_keys
        nodes.append(node)
    return nodes


BATCH_SIZE = 20  # מספר מתכונים בכל מנה של בקשות ל־Gemini
PAUSE_SECONDS = 40  # הפסקה בין מנות כדי לא לחרוג מהמכסה לדקה (ניסיון חוזר שולח שוב את כל המנה, וזה מבזבז מהמכסה היומית)
QUOTA_WAIT_SECONDS = 65  # במכסה החינמית המגבלה היא לדקה, ולכן ממתינים קצת יותר מדקה
MAX_QUOTA_WAITS = 5      # אם גם אחרי כמה המתנות עדיין יש חריגה, כנראה הגענו למכסה היומית


def mix_categories(recipes):
    """מסדרת את המתכונים לסירוגין לפי קטגוריה, כדי שגם אינדקס חלקי יכלול מתכונים מכל הקבצים."""
    by_category = {}
    for recipe in recipes:
        by_category.setdefault(recipe["category"], []).append(recipe)
    mixed = []
    for group in zip_longest(*by_category.values()):
        mixed.extend(recipe for recipe in group if recipe is not None)
    return mixed


def find_changed_recipes(collection, recipes, splitter):
    """אחרי תיקון בחילוץ או ב־recipes.json: מוצאת מתכונים שהטקסט שלהם באינדקס שונה מהטקסט הנוכחי,
    ומתכונים חדשים שהופרדו מהם (שורת הכותרת שלהם הופיעה בתוך ה־Chunks הישנים). כך מעדכנים רק אותם."""
    stored = collection.get(include=["documents", "metadatas"])
    old_chunks = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"]):
        old_chunks.setdefault(metadata["recipe_id"], []).append((metadata["chunk_index"], text))
    by_id = {r["recipe_id"]: r for r in recipes}
    changed_ids, old_lines = [], set()
    for recipe_id, chunks in old_chunks.items():
        current = [n.text for n in recipe_to_nodes(by_id[recipe_id], splitter)] if recipe_id in by_id else None
        if current != [text for _, text in sorted(chunks)]:
            changed_ids.append(recipe_id)
            old_lines |= {line.strip() for _, text in chunks for line in text.split("\n")}
    split_off = [r for r in recipes if r["recipe_id"] not in old_chunks
                 and r["original_text"].split("\n")[0].strip() in old_lines]
    return changed_ids, split_off


def build_index(rebuild=False, max_chunks=None, update=False):
    """בונה את האינדקס מתוך recipes.json ושומרת אותו ב־ChromaDB.
    מתכונים שכבר נמצאים באינדקס מדולגים, כך שאם הבנייה נעצרה (למשל בגלל מכסה) אפשר פשוט להריץ שוב.
    rebuild=True מוחק את האינדקס ובונה מחדש, למשל אחרי תיקון ידני של recipes.json.
    max_chunks מגביל את מספר ה־Chunks שיישלחו עכשיו, כי המכסה היומית החינמית היא 1,000 בקשות.
    update=True מעדכן רק מתכונים שהשתנו ומתכונים שהופרדו מהם, במקום לבנות הכול מחדש."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    if not RECIPES_FILE.exists():
        print("לא נמצא data/recipes.json. הריצי קודם: python extract_recipes.py")
        return
    recipes = json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    recipes = [r for r in recipes if r["extraction_status"] != "נכשל"]

    setup_embedding()
    client, collection, vector_store = get_vector_store()
    if rebuild:
        client.delete_collection(COLLECTION_NAME)
        client, collection, vector_store = get_vector_store()

    # אילו מתכונים כבר נמצאים באינדקס
    existing = {m["recipe_id"] for m in collection.get(include=["metadatas"])["metadatas"]}
    todo = [r for r in recipes if r["recipe_id"] not in existing]

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                paragraph_separator="\n", separator=" ")
    if update:
        changed_ids, split_off = find_changed_recipes(collection, recipes, splitter)
        if changed_ids:
            collection.delete(where={"recipe_id": {"$in": changed_ids}})  # ה־Chunks הישנים שלהם
        todo = [r for r in recipes if r["recipe_id"] in changed_ids] + split_off
        print(f"עדכון: {len(changed_ids)} מתכונים השתנו, ו־{len(split_off)} מתכונים חדשים הופרדו מהם.")
    if max_chunks:
        selected, total = [], 0
        for recipe in mix_categories(todo):
            chunk_count = len(recipe_to_nodes(recipe, splitter))
            if total + chunk_count > max_chunks:
                break
            selected.append(recipe)
            total += chunk_count
        todo = selected
    print(f"{len(recipes)} מתכונים תקינים, {len(existing)} כבר באינדקס, {len(todo)} לאינדוקס עכשיו.")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context)

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        nodes = [node for recipe in batch for node in recipe_to_nodes(recipe, splitter)]
        for attempt in range(1, MAX_QUOTA_WAITS + 2):
            try:
                index.insert_nodes(nodes)  # כאן LlamaIndex מחשבת את ה־Embeddings ושומרת ב־ChromaDB
                break
            except Exception as error:
                # מכסה לדקה: ממתינים ומנסים שוב את אותה מנה. כל שגיאה אחרת עוצרת את הבנייה
                if is_quota_error(error) and attempt <= MAX_QUOTA_WAITS:
                    print(f"הגענו למכסת הבקשות לדקה. ממתין {QUOTA_WAIT_SECONDS} שניות (ניסיון {attempt} מתוך {MAX_QUOTA_WAITS})...")
                    time.sleep(QUOTA_WAIT_SECONDS)
                    continue
                print(f"\nהבנייה נעצרה: {describe_api_error(error)}")
                print("מה שנשמר עד עכשיו נשאר באינדקס. הריצי שוב את אותה פקודה כדי להמשיך.")
                return
        print(f"נשמרו {min(start + BATCH_SIZE, len(todo))} מתוך {len(todo)} מתכונים ({len(nodes)} Chunks במנה הזו)")
        if start + BATCH_SIZE < len(todo):
            time.sleep(PAUSE_SECONDS)

    print(f"\nהאינדקס מוכן: {collection.count()} Chunks בתיקייה {CHROMA_DIR}")


def is_quota_error(error):
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def describe_api_error(error):
    """הודעה ברורה בעברית לשגיאות נפוצות של Gemini, בלי להציג פרטים פנימיים או מפתחות."""
    text = str(error)
    if is_quota_error(error):
        return "חריגה ממכסת השימוש של Gemini. כדאי להמתין כמה דקות (או עד מחר, אם זו המכסה היומית)."
    if "API key" in text or "401" in text or "403" in text:
        return "מפתח ה־API לא תקין. בדקי את GOOGLE_API_KEY בקובץ .env."
    if "503" in text or "UNAVAILABLE" in text:
        return "מודל Gemini עמוס כרגע אצל Google. כדאי לנסות שוב בעוד כמה דקות."
    if "timeout" in text.lower() or "Timeout" in type(error).__name__:
        return "Gemini לא ענה בזמן (יותר מ־30 שניות). כדאי לנסות שוב בעוד כמה דקות."
    if "SSL" in text or "Connect" in type(error).__name__ or "connect" in text.lower():
        return "בעיית חיבור לאינטרנט או ל־Google."
    return f"שגיאה לא צפויה ({type(error).__name__})"


def show_sample_chunks(count=4):
    """מציגה כמה Chunks לדוגמה מתוך ChromaDB, עם ה־Metadata שלהם, כדי לבדוק מה נשמר בפועל."""
    _, collection, _ = get_vector_store()
    total = collection.count()
    if total == 0:
        print("האינדקס ריק. הריצי קודם: python index_recipes.py")
        return
    print(f"באינדקס יש {total} Chunks. דוגמאות:\n")
    # דוגמאות מפוזרות לאורך האינדקס, ולא רק מתחילתו
    step = max(total // count, 1)
    for offset in range(0, step * count, step):
        result = collection.get(limit=1, offset=offset, include=["documents", "metadatas"])
        metadata = result["metadatas"][0]
        print("-" * 50)
        for key in ["recipe_id", "recipe_title", "source_file", "category", "author", "chunk_index"]:
            print(f"{key}: {metadata.get(key)}")
        print(f"\n{result['documents'][0][:400]}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--show" in sys.argv:
        show_sample_chunks()
        sys.exit()
    try:
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        build_index(rebuild="--rebuild" in sys.argv, max_chunks=limit, update="--update" in sys.argv)
    except RuntimeError as error:  # למשל מפתח חסר
        print(error)
