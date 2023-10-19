import diffusers

class Inference:
    def __init__(self) -> None:
        self.base_ckpt_dir = "./models/Stable-diffusion"
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
            "image"
            ]
    
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

        pipeline_name = f"{arg_dict['purpose']}_xl" if arg_dict["is_xl"] else arg_dict["purpose"] 
        
        pipeline = self.pipeline_dict[pipeline_name]
        pipe = pipeline.from_single_file(
            pretrained_model_link_or_path=f"{self.base_ckpt_dir}/{arg_dict['model_id']}.safetensors",
            torch_dtype=arg_dict["torch_dtype"],
            # local_files_only=arg_dict["local_files_only"], # TODO: Model cache?
            resume_download=True,
            use_safetensors=arg_dict["use_safetensors"],
            scheduler=self.scheduler_dict[arg_dict["scheduler"]],
        )
        pipe.safety_checker = None

        pipe_arg_dict = {arg: arg_dict[arg] for arg in self.required_arg_list} # TODO: Get image from s3
        image_list = pipe.to("cuda")(**pipe_arg_dict).images
        arg_dict.update(pipe_arg_dict)

        for index, image in enumerate(image_list):
            image.save(f"{self.output_dir}/{index}.png") # TODO: Sending image via s3
        
        del arg_dict["torch_dtype"]
        if type(arg_dict["image"]) != str: del arg_dict["image"] # TODO: Find better way to handle
        print(arg_dict)
        
        return arg_dict
    