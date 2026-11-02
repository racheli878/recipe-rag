"""
בדיקה זמנית: כמה זמן לוקח כל שלב בשאלה אחת, ואיזה שלב נכשל.
הרצה: python check_speed.py
בדיקת מודל שפה אחר בלי לשנות את .env: python check_speed.py gemini-3.5-flash
"""

import os
import sys
import time
import traceback

import rag
from llama_index.core import Settings

sys.stdout.reconfigure(encoding="utf-8")


def timed(name, action):
    """מריצה שלב אחד ומדפיסה כמה זמן לקח, או את השגיאה המלאה."""
    start = time.time()
    try:
        result = action()
        print(f"[{time.time() - start:.1f} שניות] {name}: הצליח")
        return result
    except Exception as error:
        print(f"[{time.time() - start:.1f} שניות] {name}: נכשל\n    {type(error).__name__}: {str(error)}")
        if isinstance(error, TypeError):  # שגיאה בקוד ולא ב־Gemini: מציגים איפה בדיוק
            traceback.print_exc()


rag.setup_embedding()
if len(sys.argv) > 1:
    os.environ["GEMINI_MODEL"] = sys.argv[1]
print("מודל השפה:", os.getenv("GEMINI_MODEL"))
timed("הגדרת מודל השפה", rag.setup_llm)
timed("שאלה פשוטה ל־Gemini", lambda: Settings.llm.complete("ענה במילה אחת: מה צבע השמיים?"))
timed("זיהוי דרישות (structured_predict)",
      lambda: Settings.llm.structured_predict(
          rag.Requirements, rag.REQUIREMENTS_PROMPT,
          categories="עוגות", shown="", question="עוגת שוקולד בלי מרגרינה"))
timed("Embedding לשאלה", lambda: Settings.embed_model.get_query_embedding("עוגת שוקולד"))
