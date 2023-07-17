# automatic1111-bulk-upscale
This script utilizes the [Stable Diffusion API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) within a local copy of SD using the [AUTOMATIC1111](https://github.com/AUTOMATIC1111/) WebUI.  
What this script does, is looping over all the images in the `./image_input` folder and requesting the image meta-data using a given API endpoint. Once the generation data for the specific image is given, it'll use this data and generate a new image using a specified upscaler and the image generation data (seed, model, etc.)

## Installation
1. Install the requirements using `pip install -r requirements.txt`
2. Make sure you're starting your AUTOMATIC1111 instance with enabled endpoints (See first bulletpoint [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)).
3. Place the images generated by SD in the `image_input` folder.
4. (Optional) Specify your upscaler options in [script.py:L44](./script.py#L44).
5. Run script using `python script.py` and wait until it finishes.
6. Your upscaled images will be written to the `image_output` folder.