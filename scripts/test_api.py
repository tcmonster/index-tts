from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TestReq(BaseModel):
    msg: str

@app.post("/test")
def test(req: TestReq):
    return {"echo": req.msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)