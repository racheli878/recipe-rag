"""
ממשק Streamlit לצ'אט המתכונים.
הרצה: streamlit run app.py
"""

import streamlit as st

import rag
from index_recipes import describe_api_error


PAGE_STYLE = """
<style>
:root {
    --accent: #F2643D;
    --accent-2: #F6A13A;
    --ink: #1E1B2E;
    --muted: #6E6A7C;
    --card: #FFFFFF;
    --line: #EEE9F3;
}

/* כיווניות ורקע (הגופן מוגדר ב־.streamlit/config.toml) */
.stApp, .stChatMessage, .stMarkdown, .stChatInput textarea, .stAlert, [data-testid="stStatusWidget"] {
    direction: rtl;
    text-align: right;
}
.stApp {
    background:
        radial-gradient(900px 500px at 100% -10%, #FFE3D6 0%, transparent 60%),
        radial-gradient(700px 500px at -10% 10%, #EDE7FF 0%, transparent 55%),
        #FBFAFD;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 7rem; }

/* כותרת */
.hero { text-align: center; margin-bottom: 2rem; animation: fadeUp .5s ease both; }
.hero .pill {
    display: inline-block; padding: .3rem .9rem; border-radius: 999px;
    background: rgba(242, 100, 61, .1); color: var(--accent); font-weight: 500; font-size: .85rem;
}
.hero h1 {
    font-size: 3rem; font-weight: 800; margin: .6rem 0 .3rem; line-height: 1.1; padding: 0;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0 auto; max-width: 560px; }
.stats { display: flex; justify-content: center; gap: .6rem; margin-top: 1rem; flex-wrap: wrap; }
.stats span {
    background: var(--card); border: 1px solid var(--line); border-radius: 999px;
    padding: .3rem .9rem; font-size: .85rem; color: var(--ink); box-shadow: 0 2px 8px rgba(30, 27, 46, .04);
}

/* מסך פתיחה: כרטיסי דוגמה */
.examples { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; animation: fadeUp .6s ease both; }
.examples div {
    background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem;
    color: var(--ink); font-size: .95rem; box-shadow: 0 6px 20px rgba(30, 27, 46, .05);
}
.examples b { display: block; font-size: 1.4rem; margin-bottom: .3rem; }
.examples-title { text-align: center; color: var(--muted); font-size: .95rem; font-weight: 500; margin-bottom: .7rem; }
.examples i { font-style: normal; color: var(--muted); }
@media (max-width: 640px) { .examples { grid-template-columns: 1fr; } }

/* הודעות הצ'אט */
[data-testid="stChatMessage"] {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 1rem 1.1rem; margin-bottom: .8rem; box-shadow: 0 8px 24px rgba(30, 27, 46, .06);
    animation: fadeUp .35s ease both;
}
/* הודעות המשתמשת: בועה צבעונית וצרה יותר */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: linear-gradient(135deg, #FFE9E0, #FFF4E6); border: none; box-shadow: none;
    width: fit-content; max-width: 85%; margin-right: 0; margin-left: auto;
}
[data-testid="stChatMessage"] > div[aria-hidden="true"] {
    background: #FFF; border-radius: 12px; box-shadow: 0 2px 8px rgba(30, 27, 46, .08);
}

/* מקורות וכפתורים */
.sources-title {
    font-weight: 600; color: var(--muted); font-size: .85rem; letter-spacing: .02em;
    margin-top: .8rem; padding-top: .8rem; border-top: 1px dashed var(--line);
}
.stButton button {
    border-radius: 999px; border: 1px solid transparent; font-size: .82rem; padding: .2rem .9rem;
    background: linear-gradient(#fff, #fff) padding-box, linear-gradient(90deg, var(--accent), var(--accent-2)) border-box;
    color: var(--accent); transition: all .2s ease;
}
.stButton button:hover {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) border-box; color: #fff;
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(242, 100, 61, .25);
}

/* שדה השאלה */
[data-testid="stChatInput"] > div {
    border-radius: 999px !important; border: 1px solid var(--line) !important; background: #FFF !important;
    box-shadow: 0 10px 30px rgba(30, 27, 46, .10);
}
[data-testid="stChatInput"] button { border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); color: #fff; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
"""

AVATARS = {"user": "💬", "assistant": "🍰"}


def setup_page():
    """הגדרות הדף והעיצוב: גופן עברי, RTL, צבעים, ובועות צ'אט."""
    st.set_page_config(page_title="חיפוש מתכונים", page_icon="🍰", layout="centered")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)


def show_header(recipes):
    """כותרת הדף, עם מספר המתכונים והקטגוריות במאגר."""
    categories = {r["category"] for r in recipes.values()}
    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">✨ עוזרת מתכונים חכמה</span>
            <h1>מה מבשלים היום?</h1>
            <p>שאלי בשפה חופשית ואמצא לך מתכונים מתוך המאגר. כל תשובה מבוססת רק על המתכונים שבמאגר, עם מקור לכל מתכון.</p>
            <div class="stats"><span>📖 {len(recipes):,} מתכונים</span><span>🗂️ {len(categories)} קטגוריות</span><span>🔎 חיפוש לפי רכיבים</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_welcome():
    """מוצגת כשעוד אין הודעות בשיחה: שלוש דוגמאות לשאלות."""
    st.markdown(
        """
        <div class="examples-title">💡 לא יודעת מה לשאול? הנה כמה דוגמאות לשאלות:</div>
        <div class="examples">
            <div><b>🍫</b><i>"</i>עוגת שוקולד בלי מרגרינה<i>"</i></div>
            <div><b>🥗</b><i>"</i>איזה סלטים מתאימים לאירוח?<i>"</i></div>
            <div><b>🍎</b><i>"</i>איזה מתכונים יש עם תפוחים?<i>"</i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="טוען את מאגר המתכונים...")
def load_system():
    """טוענת פעם אחת את האינדקס, מודל השפה והמתכונים. st.cache_resource שומר אותם בין שאלות,
    כך שהאינדקס לא נטען מחדש בכל לחיצה."""
    index = rag.load_index()
    rag.setup_llm()
    recipes = rag.load_recipes()
    return index, recipes


def show_full_recipe(recipe_id):
    """נקראת בלחיצה על "הצג מתכון מלא": מוסיפה לשיחה את המתכון המלא מ־recipes.json, בלי קריאה ל־Gemini."""
    _, recipes = load_system()
    st.session_state.messages.append(
        {"role": "assistant", "content": rag.format_full_recipe(recipes[recipe_id]), "sources": []})


def show_message(number, message):
    """מציגה הודעה אחת בצ'אט. בהודעת מערכת מוצגים גם המקורות, ולכל מקור כפתור "הצג מתכון מלא"."""
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown('<div class="sources-title">📚 המקורות</div>', unsafe_allow_html=True)
            for source in message["sources"]:
                col_text, col_button = st.columns([3, 1], vertical_alignment="center")
                col_text.markdown(f"• {source['label']}")
                col_button.button("הצג מתכון מלא", key=f"full_{number}_{source['recipe_id']}",
                                  on_click=show_full_recipe, args=(source["recipe_id"],))


def handle_question(question):
    """שולחת שאלה חדשה ל־rag.answer_question, ומציגה את שלבי העבודה האמיתיים בזמן שהם קורים."""
    index, recipes = load_system()
    history = [("משתמשת" if m["role"] == "user" else "מערכת", m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.status("מבין את הבקשה...", expanded=False) as status:
        try:
            result = rag.answer_question(index, recipes, question, history[-6:], st.session_state.shown_ids,
                                         progress=lambda step: status.update(label=step))
            status.update(label="התשובה מוכנה", state="complete")
        except Exception as error:
            status.update(label="אירעה שגיאה", state="error")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"מצטערת, לא הצלחתי לענות: {describe_api_error(error)}"})
            return

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]})
    if result["recipe_ids"]:
        st.session_state.shown_ids = result["recipe_ids"]


def main():
    setup_page()
    try:
        _, recipes = load_system()
    except Exception as error:  # למשל אינדקס שעוד לא נבנה, או מפתח חסר
        st.error(str(error) if isinstance(error, RuntimeError) else describe_api_error(error))
        st.stop()
    show_header(recipes)

    # היסטוריית השיחה נשמרת רק לאורך הסשן הנוכחי, בלי מסד נתונים
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("shown_ids", [])

    if not st.session_state.messages:
        show_welcome()
    for number, message in enumerate(st.session_state.messages):
        show_message(number, message)

    question = st.chat_input("למשל: אני מחפשת עוגת שוקולד בלי מרגרינה")
    if question:
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)
        handle_question(question)
        st.rerun()  # הצגה מחדש של כל השיחה, כולל התשובה החדשה וכפתורי המתכון המלא


main()
