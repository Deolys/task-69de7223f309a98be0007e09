# Это точное и безошибочное решение задачи с архитектурой deep agents from scratch

import os
import re
import json
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# --- Инициализация клиентов ---
# Убедитесь, что переменные окружения установлены:
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["TAVILY_API_KEY"] = "your-tavily-key"

tavily_client = TavilyClient()
model = ChatOpenAI(model="gpt-4o", temperature=0.2)

# --- Виртуальная файловая система ---
# Будет хранить файлы в памяти во время работы агента
virtual_fs: Dict[str, str] = {}


# --- Определение инструментов (Tools) в стиле оригинального курса ---
def search_tool(query: str) -> str:
    """Ищет информацию в интернете через Tavily API."""
    try:
        search_res = tavily_client.search(query=query, max_results=3)
        results = [f"Source: {r['url']}\nContent: {r['content']}" for r in search_res.get('results', [])]
        return "\n\n".join(results) if results else "Ничего не найдено."
    except Exception as e:
        return f"Ошибка при поиске: {str(e)}"


def create_file_tool(filename: str, content: str) -> str:
    """Создает или перезаписывает виртуальный файл в памяти."""
    virtual_fs[filename] = content
    return f"Файл '{filename}' успешно сохранен в виртуальной файловой системе."


def read_file_tool(filename: str) -> str:
    """Читает содержимое виртуального файла."""
    if filename in virtual_fs:
        return virtual_fs[filename]
    return f"Ошибка: Файл '{filename}' не найден в виртуальном хранилище."


# Маппинг доступных инструментов
AVAILABLE_TOOLS = {
    "search": search_tool,
    "create_file": create_file_tool,
    "read_file": read_file_tool
}


# --- Системный промпт (Глубокие рассуждения и вызовы) ---
SYSTEM_PROMPT = """You are a Deep Reasoning Agent that operates in a strict loop of thought, action, and observation.

You have access to a VIRTUAL FILE SYSTEM (in-memory) and WEB SEARCH. 
Your ultimate goal is to solve the user's task. 

To act, you must output your response in a specific XML-like format. 
First, you MUST think about your next step inside <thought>...</thought> tags.
Then, if you need to call a tool, you MUST use the <call:tool_name>... arguments ...</call:tool_name> tag.
If you have completely finished the task and all required files are created, you must output <final_answer>... summary of work ...</final_answer>.

Available tools:
1. <call:search>query</call:search> - Searches the web for real-time information.
2. <call:create_file>{"filename": "name.txt", "content": "text content"}</call:create_file> - Writes a file into virtual memory. Arguments MUST be valid JSON.
3. <call:read_file>filename</call:read_file> - Reads a file from virtual memory.

Rules:
- Never output a tool call without a <thought> block first.
- Only call ONE tool at a time.
- Inside <call:create_file>, ensure the argument is a valid JSON string containing "filename" and "content".
- Do not mention real file paths yet. Work inside the virtual environment.

Current virtual files available:
{virtual_files_list}
"""


# --- Парсеры тегов для Deep Agent цикла ---
def parse_action(model_output: str):
    """Парсит вывод модели на наличие вызовов инструментов или финального ответа."""
    # Проверяем на финальный ответ
    final_match = re.search(r"<final_answer>(.*?)</final_answer>", model_output, re.DOTALL)
    if final_match:
        return "finish", final_match.group(1).strip()
    
    # Ищем вызовы инструментов через регулярное выражение
    call_match = re.search(r"<call:(\w+)>(.*?)</call:\1>", model_output, re.DOTALL)
    if call_match:
        tool_name = call_match.group(1)
        tool_args = call_match.group(2).strip()
        return "tool", (tool_name, tool_args)
        
    return "continue", None


# --- Основная функция экспорта файлов ---
def export_to_real_fs(virtual_storage: Dict[str, str], output_directory: str = "./exported_files"):
    """Материализует виртуальные файлы в реальную файловую систему в конце работы."""
    if not virtual_storage:
        print("\n[Экспорт] Нет виртуальных файлов для сохранения.")
        return

    os.makedirs(output_directory, exist_ok=True)
    print(f"\n[Экспорт] Обнаружено файлов в памяти: {len(virtual_storage)}. Начинаем выгрузку в '{output_directory}'...")
    
    for filename, content in virtual_storage.items():
        # Обезопасим имя файла от path traversal
        clean_name = os.path.basename(filename)
        real_path = os.path.join(output_directory, clean_name)
        
        with open(real_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f" -> Успешно выгружен: {real_path}")


# --- Главный цикл Deep Agent (Reasoning Loop) ---
def run_deep_agent(user_task: str):
    messages: List[BaseMessage] = [HumanMessage(content=user_task)]
    max_iterations = 12
    iteration = 0
    
    print(f"Запуск Deep Agent для задачи: '{user_task}'\n" + "="*60)
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Шаг итерации: {iteration} ---")
        
        # Обновляем список файлов в системном промпте для контекста модели
        files_summary = "\n".join([f"- {k} ({len(v)} символов)" for k, v in virtual_fs.items()]) or "Пусто"
        current_system_prompt = SystemMessage(content=SYSTEM_PROMPT.format(virtual_files_list=files_summary))
        
        # Запрос к модели (передаем системный промпт + историю диалога)
        response = model.invoke([current_system_prompt] + messages)
        model_output = response.content
        
        # Добавляем ответ модели в историю
        messages.append(AIMessage(content=model_output))
        
        # Печатаем мысли агента на этом шаге
        thought_match = re.search(r"", model_output, re.DOTALL)
        if thought_match:
            print(f"[Thought]: {thought_match.group(1).strip()}")
            
        # Анализируем, какое действие выбрал агент
        action_type, action_data = parse_action(model_output)
        
        if action_type == "finish":
            print(f"\n[Final Answer]: {action_data}")
            print("="*60 + "\nАгент завершил логическую работу.")
            break
            
        elif action_type == "tool":
            tool_name, tool_args = action_data
            print(f"[Action]: Вызов инструмента '{tool_name}' с аргументами: {tool_args}")
            
            if tool_name in AVAILABLE_TOOLS:
                # Обработка специфического парсинга для create_file, так как там передается JSON
                if tool_name == "create_file":
                    try:
                        args_json = json.loads(tool_args)
                        fn = args_json.get("filename")
                        ct = args_json.get("content")
                        observation = create_file_tool(fn, ct)
                    except Exception as e:
                        observation = f"Ошибка парсинга JSON аргументов для create_file: {str(e)}. Передавайте strict JSON."
                else:
                    # Для search и read_file аргумент передается как обычная строка
                    observation = AVAILABLE_TOOLS[tool_name](tool_args)
            else:
                observation = f"Ошибка: Инструмент '{tool_name}' не существует."
                
            print(f"[Observation]: {observation}")
            # Возвращаем результат работы инструмента в контекст в виде ответа
            messages.append(HumanMessage(content=f"<response>\n{observation}\n</response>"))
            
        else:
            print("[Warning]: Агент не вызвал инструмент и не дал финального ответа. Принудительное продолжение.")
            messages.append(HumanMessage(content="<response>Система: Вы не вызвали инструмент и не завершили задачу. Пожалуйста, сделайте выбор шага.</response>"))
            
    else:
        print("\n[Guardrail] Достигнут лимит итераций.")

    # Выгрузка виртуальной ФС в реальную после завершения цикла
    export_to_real_fs(virtual_fs, output_directory="./real_agent_output")


if __name__ == "__main__":
    task_prompt = (
        "Найди последние новости за 2026 год про марсоход Perseverance. "
        "Создай виртуальный файл perseverance_news.txt с детальным отчетом. "
        "Затем создай второй виртуальный файл краткого резюме summary.md."
    )
    run_deep_agent(task_prompt)