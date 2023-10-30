import uvicorn
from fastapi import FastAPI, BackgroundTasks, APIRouter
from starlette.middleware.cors import CORSMiddleware
from diffusion.inference import Inference


class InferenceRequest:
    def __init__(self):
        self.log_path = "logs"
        
        self.infer = Inference()
        self.router = APIRouter()
        self.router.add_api_route("/api/img2img", self.img2img, methods=["POST"])
        

    async def img2img(self, request_body: dict, background_tasks: BackgroundTasks):
        request_body["purpose"] = "img2img"
        data = self.infer.run_inference(request_body)

        return data


if __name__ == "__main__":
    app = FastAPI()

    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    req = InferenceRequest()
    app.include_router(req.router)
    
    uvicorn.run(app, host="0.0.0.0", port=3000)