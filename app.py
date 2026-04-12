from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from database import (init_db, create_user, get_user_by_username, get_user_by_api_key,
                      add_document, get_user_documents, verify_password,
                      create_session, get_user_by_session, delete_session)
from rag import process_and_store_document, answer_query
from inference import openenv_reset, openenv_validate, grade_task1, grade_task2, grade_task3, GRADER_REGISTRY

load_dotenv()
app = FastAPI(title="Devkey AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

init_db()

# GRADED_TASKS must reference named grader functions (not lambdas)
# so the OpenEnv validator can match them to openenv.yaml grader names
GRADED_TASKS = {
    "task1": {
        "name": "document_qa",
        "description": "Answer a question based on uploaded documents",
        "grader": grade_task1,
        "grader_name": "grade_task1",
    },
    "task2": {
        "name": "summarization",
        "description": "Summarize the content of an uploaded document",
        "grader": grade_task2,
        "grader_name": "grade_task2",
    },
    "task3": {
        "name": "keyword_extraction",
        "description": "Extract key topics from a document",
        "grader": grade_task3,
        "grader_name": "grade_task3",
    },
}
_env_state = {"task_index": 0, "tasks": list(GRADED_TASKS.keys()), "done": False}

def get_current_user_from_cookie(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    return get_user_by_session(token)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, error: str = None):
    user = get_current_user_from_cookie(request)
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "error": error})

@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    user_id, api_key = create_user(username, password)
    if not user_id:
        return RedirectResponse(url="/?error=Username already taken", status_code=303)
    session_token = create_session(user_id)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session", value=session_token, httponly=True, samesite="none", secure=True)
    return response

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        return RedirectResponse(url="/?error=Invalid credentials", status_code=303)
    session_token = create_session(user["id"])
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session", value=session_token, httponly=True, samesite="none", secure=True)
    return response

@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        delete_session(token)
    response = RedirectResponse(url="/")
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, error: str = None):
    user = get_current_user_from_cookie(request)
    if not user:
        return RedirectResponse(url="/")
    docs = get_user_documents(user["id"])
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"request": request, "user": user, "documents": docs, "error": error})

@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    user = get_current_user_from_cookie(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        return RedirectResponse(url="/dashboard?error=Unsupported file type", status_code=303)
    contents = await file.read()
    doc_id = add_document(user["id"], file.filename)
    try:
        process_and_store_document(user["id"], doc_id, contents, file.filename)
    except Exception as e:
        print(f"Error processing document: {e}")
        return RedirectResponse(url="/dashboard?error=Failed to process document", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)

class ChatRequest(BaseModel):
    query: str

def get_user_from_api_keyHeader(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    api_key = auth_header.split(" ")[1]
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    user = get_user_from_api_keyHeader(request)
    answer = answer_query(user["id"], req.query)
    return {"answer": answer}

@app.post("/web/chat")
async def web_chat(req: ChatRequest, request: Request):
    user = get_current_user_from_cookie(request)
    if not user:
        return JSONResponse(status_code=401, content={"answer": "Unauthorized"})
    answer = answer_query(user["id"], req.query)
    return {"answer": answer}

@app.post("/reset")
async def endpoint_reset():
    try:
        _env_state["task_index"] = 0
        _env_state["done"] = False
        result = openenv_reset()
        result["tasks"] = [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "grader": v["grader_name"],
                "enabled": True,
            }
            for k, v in GRADED_TASKS.items()
        ]
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/step")
async def endpoint_step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = body.get("action", body.get("message", str(body)))
    task_keys = list(GRADED_TASKS.keys())
    idx = _env_state.get("task_index", 0)
    if idx >= len(task_keys):
        return JSONResponse(status_code=200, content={
            "observation": "Episode complete",
            "reward": 0.5,
            "done": True,
            "info": {"task": "complete"}
        })
    task_key = task_keys[idx]
    task = GRADED_TASKS[task_key]
    reward = task["grader"](action)
    _env_state["task_index"] = idx + 1
    done = (_env_state["task_index"] >= len(task_keys))
    return JSONResponse(status_code=200, content={
        "observation": f"Completed {task['name']}: {task['description']}",
        "reward": round(reward, 4),
        "done": done,
        "info": {"task": task_key, "task_name": task["name"], "grader": task["grader_name"]}
    })

@app.get("/tasks")
async def endpoint_tasks():
    return JSONResponse(status_code=200, content={
        "tasks": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "grader": v["grader_name"],
                "enabled": True,
                "max_attempts": 5,
                "scoring": "0.0-1.0 partial credit"
            }
            for k, v in GRADED_TASKS.items()
        ]
    })

@app.get("/state")
async def endpoint_state():
    try:
        result = openenv_validate()
        result["tasks"] = [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "grader": v["grader_name"],
                "enabled": True,
            }
            for k, v in GRADED_TASKS.items()
        ]
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/openenv/reset")
async def endpoint_openenv_reset():
    return await endpoint_reset()

@app.get("/openenv/validate")
async def endpoint_openenv_validate():
    return await endpoint_state()

@app.get("/health")
async def health():
    return {"status": "ok", "graders": list(GRADER_REGISTRY.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


@app.post("/grader")
async def endpoint_grader(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = body.get("action", body.get("response", str(body)))
    task_id = body.get("task_id", "task1")
    task = GRADED_TASKS.get(task_id, list(GRADED_TASKS.values())[0])
    reward = task["grader"](action)
    return JSONResponse(status_code=200, content={"reward": round(reward, 4), "task_id": task_id, "scored": True})

