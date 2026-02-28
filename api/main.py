from fastapi import FastAPI

from api.routes.debug_frame import router as debug_frame_router
from api.routes.health import router as health_router
from api.routes.video_stream import router as video_stream_router

app = FastAPI()
app.include_router(debug_frame_router)
app.include_router(health_router)
app.include_router(video_stream_router)


@app.get("/")
async def root():
    return {"message": "ok"}
