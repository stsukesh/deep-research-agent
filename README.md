---
title: Enterprise Research Agent
emoji: 🌐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent AI research & report generation system
---

# 🌐 Enterprise Multi-Agent Research & Report Generation System

An advanced multi-agent AI research pipeline that autonomously plans, researches, extracts, writes, and evaluates professional reports. 

Built with **LangGraph**, **FastAPI**, and **Neon PostgreSQL**, this system implements key production-grade patterns: **Human-in-the-Loop approval**, **durable state persistence across restarts**, and **resilient LLM fallbacks**.

---

## 🏗️ System Architecture & Workflow

The agent orchestrates six specialized sub-agents inside a cyclic LangGraph `StateGraph`:

```
                    START
                      │
                      ▼
                   Planner ──► Researcher ──► Extractor
                                  ▲              │
                                  │              ▼
                                  └── [Reject] ── Approval (Interrupt)
                                                 │
                                                 ├── [Approve]
                                                 ▼
                                              Writer ◄───┐
                                                 │       │ [Rewrite]
                                                 ▼       │
                                              Reviewer ──┘
                                                 │
                                                 └── [Approve] ──► END
```

### 👥 The 6 Specialized Agents
1. **Planner**: Breaks the research query into target topics, scope, and objectives.
2. **Researcher**: Parallelized searches across Tavily, Wikipedia, Arxiv, and DuckDuckGo.
3. **Extractor**: Translates raw results into structured Pydantic findings with confidence ratings.
4. **Approval (Human-in-the-Loop)**: Interrupts the graph, serializes state to PostgreSQL, and waits for user feedback.
5. **Writer**: Synthesizes verified findings into a beautifully structured Markdown report.
6. **Reviewer**: Grades the report against criteria, prompting a rewrite loop if standards are not met.

---

## ✨ Advanced Production Patterns

### 1. Durable PostgreSQL State Checkpointing (Neon Cloud)
Instead of using memory-only checkpointers (which lose state if the server restarts), we use LangGraph's `AsyncPostgresSaver`. 
- State checkpoints are written to Neon cloud Postgres after *every single node execution*.
- Enables **fault-tolerant human-in-the-loop**: the server can shut down or restart while a job is `awaiting_approval`. Once the user hits the `/api/approve` endpoint, the graph resumes execution from the exact checkpoint stored in Postgres.
- Database health checking is integrated directly into the startup lifecycle.

### 2. Dual LLM Provider Fallbacks & Token Limit Optimization
To solve API rate-limits and token exhaustion:
- **Primary LLM**: **NVIDIA NIM** (`meta/llama-3.3-70b-instruct` via OpenAI compatible endpoints). It handles high-reasoning, structured extraction tasks without daily token limits.
- **Fallback LLM**: **Groq** (`llama-3.1-8b-instant` or `llama-3.3-70b-versatile` with `max_retries=0`). If NVIDIA NIM is slow or unavailable, LangChain's `RunnableWithFallbacks` transparently fails over to Groq.
- **Fast-Path Generation**: The Writer and Reviewer agents use a specialized fast-path LLM (Groq 8B) running at **500+ tokens/second** to produce large Markdown reports in under 5 seconds, rather than waiting 90 seconds on a 70B model.

### 3. Quantitative Evaluation Metrics
Every completed report records a set of metrics saved directly to the database:
- Total execution time.
- Number of tool invocations.
- Average confidence score across all extracted sources.
- Number of cited references.

---

## 📡 REST API Endpoints

FastAPI endpoints manage the pipeline asynchronously, running agents as background tasks:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/research` | Start a new research job (returns `job_id`) |
| `GET` | `/api/status/{job_id}` | Retrieve job state, progress, and findings for approval |
| `POST` | `/api/approve/{job_id}` | Approve findings or submit feedback to reject them |
| `GET` | `/api/report/{job_id}` | Retrieve the final generated markdown report |
| `GET` | `/api/jobs` | List all historical research jobs |
| `GET` | `/api/metrics/{job_id}` | Retrieve quantitative evaluation metrics |
| `GET` | `/api/health` | Check FastAPI service and Neon Postgres connection health |

---

## 🚀 Getting Started

### 1. Environment Configuration
Create a `.env` file in the root directory:
```env
# LLM Providers
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_MODEL=meta/llama-3.3-70b-instruct

GROQ_API_KEY=gsk_your-key
GROQ_MODEL=llama-3.1-8b-instant

# Search Integration
TAVILY_API_KEY=tvly-your-key

# Database Connection URLs (Neon PostgreSQL)
DATABASE_URL=postgresql+asyncpg://neondb_owner:password@host/neondb?ssl=require
CHECKPOINT_DB_URL=postgresql://neondb_owner:password@host/neondb?sslmode=require

# App Configuration
APP_ENV=development
MAX_REVISIONS=1
```

### 2. Run with Docker (Recommended)
Build and start the application in one command:
```bash
docker compose up --build -d
```

### 3. Run Locally (Development)
Ensure you have Python 3.12+ installed:
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start local server
python main.py
```

- **Web Dashboard**: [http://localhost:7860](http://localhost:7860)
- **API Documentation**: [http://localhost:7860/docs](http://localhost:7860/docs)

---

## 🧪 Testing

The repository includes a suite of isolation tests for agent nodes and integration tests for FastAPI REST controllers:

```bash
# Run tests inside docker container
docker compose exec app pytest

# Run tests locally
pytest tests/ -v
```

---

## 📂 Project Structure

```
enterprise-research-agent/
├── app/
│   ├── agents/           # Specialized agent nodes (Planner, Writer, etc.)
│   │   └── llm_factory.py# Resilient LLM config with fallback logic
│   ├── tools/            # Tavily, DDG, Wikipedia, and Arxiv integrations
│   ├── graph/            # LangGraph StateGraph definition
│   ├── api/              # FastAPI APIRouter handlers
│   ├── database/         # SQLAlchemy ORM models & session connections
│   └── evaluations/      # Citations & duration metrics collector
├── static/               # Frontend dashboard (HTML/CSS/JS)
├── tests/                # Unit & API integration tests
├── Dockerfile            # Container build specification
├── docker-compose.yml    # Service orchestration
└── main.py               # Fast API entry point and application lifespan
```

## 📝 License

This project is licensed under the MIT License.
