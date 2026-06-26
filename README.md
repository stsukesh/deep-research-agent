# 🌐 Enterprise Research & Report Generation Agent

> **Multi-agent AI research system** that autonomously plans, researches, extracts, writes, and reviews professional reports — with human-in-the-loop approval, persistent memory, and evaluation metrics.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)

## 🏗️ Architecture

```
START → Planner → Researcher → Extractor → Human Approval
                                                │
                                     ┌──────────┴──────────┐
                                     │ approved            │ rejected
                                     ▼                     ▼
                                   Writer              Researcher
                                     │                  (re-run)
                                     ▼
                                  Reviewer
                                ┌────┴────┐
                                │approved │rewrite
                                ▼         ▼
                               END      Writer
```

## ✨ Key Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Graph Orchestration** | LangGraph | Cyclic StateGraph with conditional edges |
| **6 Specialized Agents** | LangChain + Groq | Planner, Researcher, Extractor, Approval, Writer, Reviewer |
| **Tool Calling** | Tavily, Wikipedia, Arxiv, DuckDuckGo | Multi-source research |
| **Structured Output** | Pydantic | Forced JSON schema compliance |
| **Human-in-the-Loop** | LangGraph interrupt() | Pause/resume with persistent state |
| **Memory & Checkpointing** | PostgreSQL | Full state persistence across restarts |
| **Evaluation Pipeline** | Custom metrics | Time, tool calls, confidence, citations |
| **REST API** | FastAPI | Async endpoints with background tasks |
| **Web UI** | Vanilla JS | Pipeline visualization + approval workflow |
| **Containerized** | Docker Compose | One-command deployment |

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd enterprise-research-agent
cp .env.example .env
# Edit .env with your API keys
```

### 2. Add API Keys to `.env`
```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Run Locally (Development)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 5. Open the App
- **Web UI**: http://localhost:7860
- **API Docs**: http://localhost:7860/docs
- **Health Check**: http://localhost:7860/api/health

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/research` | Start a new research job |
| `GET` | `/api/status/{id}` | Check job status + findings |
| `POST` | `/api/approve/{id}` | Approve/reject findings (HIL) |
| `GET` | `/api/report/{id}` | Get the final report |
| `GET` | `/api/jobs` | List all research jobs |
| `GET` | `/api/metrics/{id}` | Get evaluation metrics |
| `GET` | `/api/health` | Health check |

### Example Usage
```bash
# Start research
curl -X POST http://localhost:7860/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Nvidia AI business strategy for 2026"}'

# Check status (returns findings when awaiting approval)
curl http://localhost:7860/api/status/{job_id}

# Approve findings
curl -X POST http://localhost:7860/api/approve/{job_id} \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'

# Get report
curl http://localhost:7860/api/report/{job_id}
```

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📂 Project Structure
```
enterprise-research-agent/
├── app/
│   ├── agents/           # 6 specialized agents
│   │   ├── planner.py    # Query → Research Plan
│   │   ├── researcher.py # Plan → Search Results
│   │   ├── extractor.py  # Results → Structured Findings
│   │   ├── approval.py   # Human-in-the-Loop (interrupt)
│   │   ├── writer.py     # Findings → Professional Report
│   │   └── reviewer.py   # QA Review → Approve/Rewrite
│   ├── tools/            # Search tool integrations
│   ├── graph/            # State design + graph builder
│   ├── schemas/          # Pydantic models (structured output)
│   ├── api/              # FastAPI routes
│   ├── database/         # SQLAlchemy models + connection
│   └── evaluations/      # Metrics tracking
├── static/               # Web UI (HTML/CSS/JS)
├── tests/                # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
└── main.py
```

## 🧠 Skills Demonstrated

- **LangGraph**: StateGraph, conditional edges, cycles, reducers
- **Agents**: 6 specialized agents with distinct responsibilities
- **Tool Calling**: Multi-tool research (Tavily, Wikipedia, Arxiv, DuckDuckGo)
- **Structured Output**: Pydantic schema enforcement via with_structured_output()
- **Human-in-the-Loop**: interrupt() + Command(resume=...) for approval workflows
- **Memory**: PostgreSQL checkpointer for persistent state
- **FastAPI**: Async REST API with background tasks
- **Docker**: Multi-service deployment with docker-compose
- **Testing**: Unit tests for agents + API endpoint tests
- **Evaluation**: Quantitative metrics pipeline

## 📝 License

MIT
