# sd-inference
Stable diffusion diffusers img2img

## How to use API
```json
{
    "model_id": "sd_xl_base_1.0_0.9vae",
    "is_xl": true,
    "image": "<s3 path>",
    "prompt": "<prompt>",
    "negative_prompt": "<negative prompt>",
    "num_inference_steps": 30,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 7,
    "num_images_per_prompt": 4,
}
```
## output
```json
{
    "model_id": string,
    "is_xl": bool,
    "image": string,
    "prompt": string,
    "negative_prompt": string,
    "num_inference_steps": int,
    "height": int,
    "width": int,
    "guidance_scale": int,
    "num_images_per_prompt": int,

    "image_path": list of string
}
