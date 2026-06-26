"""
Enterprise Research Agent — Main Entry Point
=============================================
WHAT: FastAPI application setup with CORS, static files, and router mounting.
HOW:  Creates FastAPI app, configures middleware, mounts the API router,
      and serves the static web UI.
WHY:  This is the single entry point for the entire application.
      `uvicorn main:app` starts everything.

INTERVIEW Q&A:
  Q: How do you structure a FastAPI application?
  A: I use a modular structure — routes in a separate module, mounted via
     APIRouter with a prefix. Static files are served via StaticFiles mount.
     CORS middleware enables the web UI to call the API. Lifespan events
     handle startup (DB init) and shutdown (connection cleanup).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.api import routes as _routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- Startup -----
    print("🚀 Enterprise Research Agent starting up...")
    
    from app.config import get_settings
    settings = get_settings()
    
    try:
        if settings.CHECKPOINT_DB_URL:
            print("🗄️  Connecting to Neon PostgreSQL...")
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with AsyncPostgresSaver.from_conn_string(settings.CHECKPOINT_DB_URL) as checkpointer:
                # Create LangGraph checkpoint tables if they don't exist
                await checkpointer.setup()
                await _routes.startup(checkpointer)
                print("✅ Database ready. Graph compiled with Postgres checkpointer.")
                print("📊 API docs available at http://localhost:7860/docs")
                yield
        else:
            print("⚠️  CHECKPOINT_DB_URL not found, using memory fallback...")
            await _routes.startup(None)
            print("✅ App ready with MemorySaver.")
            yield
    finally:
        # ----- Shutdown -----
        from app.database.connection import close_db
        await close_db()
        print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Enterprise Research Agent",
    description="Multi-agent AI research system with LangGraph, tool calling, structured output, human-in-the-loop, and memory.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allows the web UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api")

# Mount static files for the web UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    """Serve the web UI."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)


