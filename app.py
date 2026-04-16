from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import recommend_movies

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieRequest(BaseModel):
    genre: str
    rating: float


@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running 🚀"}

@app.post("/recommend")
def recommend(data: MovieRequest):
    movies = recommend_movies(data.genre, data.rating)
    return {"recommendations": movies}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)