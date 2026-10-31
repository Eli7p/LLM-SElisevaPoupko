import csv
import html
import io
import json
import os
import ssl
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import httpx2
from openai import OpenAI


SYSTEM_PROMPT = """
את/ה אייג'נט CLI שעוזר/ת למשתמשת לתכנן את יום העבודה שלה.

תאריך התכנון נמסר לך בהודעת המשתמשת. השתמש/י בו בכלי read_tasks.
הכיני טבלה עם מזהה המשימה, שמה, רמת הדחיפות (גבוהה, בינונית או נמוכה)
וטווח זמן משוער בדקות. ההערכות אינן ודאיות ויש לבסס אותן על תיאור המשימה.
אם חסר מידע הכרחי, השתמש/י בכלי ask_user.

לאחר קריאת המשימות והערכתן, הפעילי את print_task_table כדי להציג טבלה מסודרת במסוף.
אחר כך שאלי באמצעות ask_user אם המשתמשת רוצה לשמור את הטבלה בקובץ.
אם היא עונה בחיוב, שמרי אותה בקובצי JSON, TXT, CSV ו-HTML מעוצב,
בשמות הכוללים את תאריך התכנון.
הפעילי את create_task_file רק אחרי תשובה חיובית.
אם המשתמשת מסרבת, אל תיצרי קובץ.
דווחי על שמירה רק לאחר שהכלי create_task_file הצליח.
"""


def load_settings():
    """Load the API settings from .env without displaying their values."""
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    setting_names = ("OPENAI_API_KEY", "MODEL", "BASE_URL")
    settings = {name: os.getenv(name) for name in setting_names}
    missing_settings = [name for name, value in settings.items() if not value]

    if missing_settings:
        missing_names = ", ".join(missing_settings)
        raise ValueError(f"Missing settings in .env: {missing_names}")

    return settings


def create_client(settings):
    """Create an OpenAI client with relaxed strict X.509 checks."""
    ssl_context = ssl.create_default_context()
    ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT

    return OpenAI(
        api_key=settings["OPENAI_API_KEY"],
        base_url=settings["BASE_URL"],
        http_client=httpx2.Client(verify=ssl_context),
    )


def get_target_date():
    """Ask which date to plan for and return it as an ISO date string."""
    while True:
        answer = input(
            "לאיזה תאריך להכין את התכנון? הקישי 'היום' או תאריך בפורמט YYYY-MM-DD: "
        ).strip()
        if answer.lower() in ("היום", "today"):
            return date.today().isoformat()
        try:
            return date.fromisoformat(answer).isoformat()
        except ValueError:
            print("תאריך לא תקין. הקישי 'היום' או תאריך בפורמט YYYY-MM-DD.")


def read_tasks(target_date):
    """Read date-specific mock tasks, falling back to the default mock."""
    data_dir = Path(__file__).parent
    dated_file = data_dir / f"mock_tasks_{target_date}.json"
    default_file = data_dir / "mock_tasks.json"
    if dated_file.exists():
        file_path = dated_file
    elif default_file.exists():
        file_path = default_file
    else:
        raise FileNotFoundError(
            f"No mock task file found for {target_date}: "
            f"expected {dated_file.name} or {default_file.name}."
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ValueError("The JSON file must contain a 'tasks' list.")

    return data["tasks"]


def format_task_table(target_date, tasks):
    """Format the estimated tasks as an aligned console table."""
    headers = ("מזהה", "משימה", "דחיפות", "זמן משוער")
    rows = [
        (
            str(task["id"]),
            str(task["title"]),
            str(task["urgency"]),
            f'{task["estimate_min_minutes"]}-{task["estimate_max_minutes"]} דקות',
        )
        for task in tasks
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [f"תכנון ליום {target_date}", format_row(headers), separator]
        + [format_row(row) for row in rows]
    )


def print_task_table(target_date, tasks):
    """Print the estimated task table to the console."""
    print(f"\n{format_task_table(target_date, tasks)}")
    return "הטבלה הוצגה במסוף."


def ask_user(question):
    """Ask the user a question in the CLI and return the answer."""
    print(f"\nאייג'נט >>> {question}")
    return input("את >>> ").strip()


def create_task_file(target_date, tasks):
    """Save estimated tasks as date-named JSON, TXT, CSV, and styled HTML files."""
    output_dir = Path(__file__).parent / "exports"
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / f"tasks_{target_date}.json"
    txt_path = output_dir / f"tasks_{target_date}.txt"
    csv_path = output_dir / f"tasks_{target_date}.csv"
    html_path = output_dir / f"tasks_{target_date}.html"

    text_content = format_task_table(target_date, tasks)
    csv_buffer = io.StringIO(newline="")
    fieldnames = [
        "id",
        "title",
        "urgency",
        "estimate_min_minutes",
        "estimate_max_minutes",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(tasks)

    task_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(task['id']))}</td>"
        f"<td class=\"task-title\">{html.escape(str(task['title']))}</td>"
        f"<td><span class=\"urgency\">{html.escape(str(task['urgency']))}</span></td>"
        f"<td>{task['estimate_min_minutes']}–{task['estimate_max_minutes']} דקות</td>"
        "</tr>"
        for task in tasks
    )
    html_content = f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>תכנון יום העבודה · {html.escape(target_date)}</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; background: linear-gradient(135deg, #f3edff, #e8f8f4); color: #24213a; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 56px 24px; }}
    .card {{ overflow: hidden; background: #fff; border: 1px solid #eee9f7; border-radius: 24px; box-shadow: 0 18px 55px #33205b18; }}
    header {{ padding: 32px; background: linear-gradient(120deg, #7657d5, #a46be3); color: #fff; }}
    .eyebrow {{ margin: 0 0 8px; opacity: .8; font-size: .9rem; }}
    h1 {{ margin: 0; font-size: clamp(1.7rem, 5vw, 2.4rem); }}
    .date {{ margin: 10px 0 0; opacity: .9; }}
    .table-wrap {{ overflow-x: auto; padding: 12px 24px 24px; }}
    table {{ width: 100%; border-collapse: collapse; text-align: right; }}
    th {{ padding: 16px 12px; color: #77718d; font-size: .9rem; }}
    td {{ padding: 17px 12px; border-top: 1px solid #f0edf5; }}
    tbody tr:hover {{ background: #faf8ff; }}
    .task-title {{ min-width: 220px; font-weight: 650; }}
    .urgency {{ display: inline-block; padding: 5px 12px; border-radius: 999px; background: #f0eaff; color: #6845b8; font-weight: 600; }}
    footer {{ padding: 0 32px 26px; color: #8a859b; font-size: .85rem; }}
    @media (max-width: 600px) {{ main {{ padding: 24px 12px; }} header {{ padding: 24px; }} .table-wrap {{ padding: 8px 12px 16px; }} }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <header>
        <p class="eyebrow">✨ קצת סדר, הרבה אוויר</p>
        <h1>תכנון יום העבודה</h1>
        <p class="date">{html.escape(target_date)}</p>
      </header>
      <div class="table-wrap">
        <table>
          <thead><tr><th>מזהה</th><th>המשימה</th><th>דחיפות</th><th>זמן משוער</th></tr></thead>
          <tbody>{task_rows}</tbody>
        </table>
      </div>
      <footer>הזמנים הם הערכות ועשויים להשתנות 🌿</footer>
    </section>
  </main>
</body>
</html>
"""

    json_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(text_content, encoding="utf-8")
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8-sig")
    html_path.write_text(html_content, encoding="utf-8")

    return f"הקבצים נשמרו:\n{json_path}\n{txt_path}\n{csv_path}\n{html_path}"


read_tasks_tool = {
    "type": "function",
    "name": "read_tasks",
    "description": "קורא משימות מקובץ mock של התאריך שנבחר אם קיים, ואחרת מקובץ ברירת המחדל.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_date": {"type": "string"},
        },
        "required": ["target_date"],
        "additionalProperties": False,
    },
    "strict": True,
}

task_properties = {
    "tasks": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "urgency": {
                    "type": "string",
                    "enum": ["גבוהה", "בינונית", "נמוכה"],
                },
                "estimate_min_minutes": {"type": "integer"},
                "estimate_max_minutes": {"type": "integer"},
            },
            "required": [
                "id",
                "title",
                "urgency",
                "estimate_min_minutes",
                "estimate_max_minutes",
            ],
            "additionalProperties": False,
        },
    },
}

print_task_table_tool = {
    "type": "function",
    "name": "print_task_table",
    "description": "מדפיס למסוף טבלה מסודרת של המשימות המוערכות לתאריך שנבחר.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_date": {"type": "string"},
            **task_properties,
        },
        "required": ["target_date", "tasks"],
        "additionalProperties": False,
    },
    "strict": True,
}

ask_user_tool = {
    "type": "function",
    "name": "ask_user",
    "description": "מציג למשתמשת שאלה ומחזיר את תשובתה.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
    "strict": True,
}

create_task_file_tool = {
    "type": "function",
    "name": "create_task_file",
    "description": "שומר את טבלת המשימות בקובצי JSON, TXT, CSV ו-HTML מעוצב על שם התאריך שנבחר, אחרי אישור המשתמשת.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_date": {"type": "string"},
            **task_properties,
        },
        "required": ["target_date", "tasks"],
        "additionalProperties": False,
    },
    "strict": True,
}

tools = [read_tasks_tool, print_task_table_tool, ask_user_tool, create_task_file_tool]

TOOLS_REGISTER = {
    "read_tasks": read_tasks,
    "print_task_table": print_task_table,
    "ask_user": ask_user,
    "create_task_file": create_task_file,
}


def main():
    """Run the tool-calling conversation with a five-turn limit."""
    target_date = get_target_date()
    settings = load_settings()
    client = create_client(settings)
    history = [
        {
            "role": "user",
            "content": f"הכיני לי לוח עבודה לתאריך {target_date}.",
        }
    ]

    for _ in range(5):
        response = client.responses.create(
            model=settings["MODEL"],
            instructions=SYSTEM_PROMPT,
            input=history,
            temperature=0.2,
            tools=tools,
        )

        history.extend(response.output)
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not function_calls:
            print(f"\nאייג'נט >>>\n{response.output_text}")
            break

        for call in function_calls:
            print(f"TOOL >>> {call.name}")
            arguments = json.loads(call.arguments)
            if call.name in ("read_tasks", "print_task_table", "create_task_file"):
                arguments["target_date"] = target_date
            result = TOOLS_REGISTER[call.name](**arguments)
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        print("הגענו למגבלת 5 סבבים. אפשר להריץ שוב כדי להמשיך.")


if __name__ == "__main__":
    main()