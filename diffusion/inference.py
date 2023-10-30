import diffusers
from storage.image_storage import Boto3Client
import hashlib
import base64
import torch

class Inference:
    def __init__(self) -> None:
        self.base_ckpt_path = "./models/Stable-diffusion/<model_name>" # TODO: add model
        self.output_dir = "./output"
        
        self.pipeline_dict = {
            "img2img": diffusers.StableDiffusionImg2ImgPipeline,
            "img2img_xl": diffusers.StableDiffusionXLImg2ImgPipeline
        }
        
        self.scheduler_dict = {
            "Euler a": diffusers.EulerAncestralDiscreteScheduler(),
        }
        
        self.required_arg_list = [
            "prompt",
            "negative_prompt",
            "num_inference_steps",
            "guidance_scale",
            "num_images_per_prompt",
            "strength",
            # "image"
            ]
        
        self.boto3 = Boto3Client()

        self.pipe = diffusers.StableDiffusionXLImg2ImgPipeline.from_single_file(
            pretrained_model_link_or_path=f"{self.base_ckpt_path}.safetensors",
            torch_dtype=torch.float16,
            resume_download=True,
            use_safetensors=True,
            scheduler=diffusers.EulerAncestralDiscreteScheduler(),
        )
        self.pipe.safety_checker = None
        self.pipe.to("cuda")
    
    def init_general_arg(
        self,
        is_xl: bool = False,
        prompt: str = "",
        ):
        arg_dict = locals()
        arg_dict["prompt"] = prompt
        del arg_dict["self"]
        if is_xl:
            arg_dict["height"] = 1024
            arg_dict["width"] = 1024
            arg_dict["hires_scale"] = 1.0

        return arg_dict

        
    def run_inference(
        self,
        arg_dict: dict
        ) -> None:
        arg_dict = self.init_general_arg(**arg_dict)

        
        pipe_arg_dict = {arg: arg_dict[arg] for arg in self.required_arg_list}
        pipe_arg_dict["image"] = self.boto3.download(arg_dict["image"])

        image_list = self.pipe(**pipe_arg_dict).images
        arg_dict.update(pipe_arg_dict)

        arg_dict["image_path"] = []
        for image in image_list:
            image_hash = hashlib.md5(base64.b64decode(image)).hexdigest()
            arg_dict["image_path"].append(self.boto3.upload(image, f"{image_hash}.png"))
        
        del arg_dict["torch_dtype"]
        if type(arg_dict["image"]) != str: del arg_dict["image"] # TODO: Find better way to handle
        print(arg_dict)
        
        return arg_dict
    