from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

app = FastAPI(title="PitchQuest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # ADD THIS - Vite default port
        "http://frontend:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)