from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import gradio as gr

from journal_ai.ingestion.pdf_parser import extract_pdf_text
from journal_ai.orchestration.graph import journal_recommendation_graph
from journal_ai.ui.gradio_app import create_gradio_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    yield
    # Clean up tasks if any


app = FastAPI(
    title="AI-Based Journal Recommendation Assistant",
    description="Intelligent academic journal recommendation with LangGraph multi-agent orchestration, RAG, and real-time APIs.",
    version="1.2.0",
    lifespan=lifespan,
)


class RecommendationRequest(BaseModel):
    paper_text: str = Field(min_length=20, description="Manuscript title, abstract, or full text draft")
    preferences: dict[str, float] = Field(
        default_factory=lambda: {
            "scope": 0.25,
            "similarity": 0.20,
            "credibility": 0.20,
            "cost": 0.10,
            "impact": 0.15,
            "turnaround": 0.10,
        },
        description="Importance weights for the specialist evaluation dimensions",
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "orchestration": "LangGraph v1.2",
        "parallel_agents": 6,
        "apis": ["OpenAlex", "Crossref", "DOAJ"],
        "vector_store": "ChromaDB",
    }


@app.post("/api/recommend")
async def recommend(request: RecommendationRequest):
    result = await journal_recommendation_graph.ainvoke({
        "paper_text": request.paper_text,
        "preferences": request.preferences,
    })

    return {
        "paper_profile": result.get("paper_profile", {}),
        "recommendations": result.get("recommendations", []),
        "conflicts": result.get("conflicts", []),
        "candidate_count": len(result.get("candidate_journals", [])),
    }


@app.post("/api/recommend/upload")
async def recommend_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF manuscript file.",
        )

    pdf_bytes = await file.read()
    try:
        paper_text = extract_pdf_text(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse PDF: {exc}",
        ) from exc

    result = await journal_recommendation_graph.ainvoke({
        "paper_text": paper_text,
        "preferences": {
            "scope": 0.25,
            "similarity": 0.20,
            "credibility": 0.20,
            "cost": 0.10,
            "impact": 0.15,
            "turnaround": 0.10,
        },
    })

    return {
        "filename": file.filename,
        "paper_profile": result.get("paper_profile", {}),
        "recommendations": result.get("recommendations", []),
        "conflicts": result.get("conflicts", []),
        "candidate_count": len(result.get("candidate_journals", [])),
    }


# Mount the Interactive Gradio Interface onto FastAPI
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")


@app.get("/")
async def root_redirect():
    """Redirect root path to interactive Gradio interface."""
    return RedirectResponse(url="/gradio")


if __name__ == "__main__":
    import uvicorn
    from journal_ai.config.settings import settings

    uvicorn.run(
        "journal_ai.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
