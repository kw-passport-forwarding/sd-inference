# sd-inference
Stable diffusion diffusers img2img

## How to use API
```json
{
    "model_id": "sd_xl_base_1.0_0.9vae",
    "is_xl": true,
    "image": "s3 url", // TODO: S3 attatch
    "prompt": "<prompt>",
    "negative_prompt": "<negative prompt>",
    "num_inference_steps": 30,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 7,
    "num_images_per_prompt": 4,
}
```