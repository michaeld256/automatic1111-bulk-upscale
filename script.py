import requests
import base64
import os
import re
import traceback
import time

# CONSTANTS
STABLE_DIFFUSION_URL = "http://127.0.0.1:7860"

imageId = 1
images = os.listdir("./image_input")
os.mkdir(f"./image_output/{time.time()}")

# Iterate over every file in image_input
for inputImgName in images:
    try:
        print(f'[>] Processing image_input/{inputImgName} ({imageId}/{len(images)})')

        # Convert image to base64 string
        imageB64 = base64.b64encode(open("./image_input/" + inputImgName, "rb").read())

        # Request png info endpoint using base64 image
        responseData = {}
        responsePngInfo = requests.post(url=f'{STABLE_DIFFUSION_URL}/sdapi/v1/png-info', json={
            "image": "data:image/png;base64," + imageB64.decode()
        }).json()["info"]

        # Parse data from png-info using regular expressions and grouping
        responseData["prompt"] = re.search("(.*)\n.*Negative prompt:", responsePngInfo, re.DOTALL)
        responseData["negative_prompt"] = re.search("Negative prompt: (.*?)Steps:", responsePngInfo, re.DOTALL)
        responseData["steps"] = re.search("Steps: (.*?)(?:,|$)", responsePngInfo)
        responseData["sampler_name"] = re.search("Sampler: (.*?)(?:,|$)", responsePngInfo)
        responseData["cfg_scale"] = re.search("CFG scale: (.*?)(?:,|$)", responsePngInfo)
        responseData["seed"] = re.search("Seed: (.*?)(?:,|$)", responsePngInfo)
        responseData["size"] = re.search("Size: (.*?)(?:,|$)", responsePngInfo)
        responseData["modelhash"] = re.search("Model hash: (.*?)(?:,|$)", responsePngInfo)
        responseData["model"] = re.search("Model: (.*?)(?:,|$)", responsePngInfo)
        responseData["subseed"] = re.search("Variation seed: (.*?)(?:,|$)", responsePngInfo)
        responseData["subseed_strength"] = re.search("Variation seed strength: (.*?)(?:,|$)", responsePngInfo)
        print(f'    [+] Got PNG-Info!')

        # prepare payload for txt2img request
        payload = {
            "enable_hr": True,
            "denoising_strength": 0.4,
            "hr_scale": 2,
            "hr_upscaler": "R-ESRGAN 4x+",
            "hr_second_pass_steps": 20,
        }

        # Add parsed values to payload
        for key, value in responseData.items():
            if type(value) is not re.Match:
                continue

            payload[key] = value[1]

        # Fix some values from payload
        payload["width"] = payload["size"].split("x")[0] if "size" in payload else 512
        payload["height"] = payload["size"].split("x")[1] if "size" in payload else 512
        payload["steps"] = int(payload["steps"]) if "steps" in payload else 50
        payload["cfg_scale"] = int(payload["cfg_scale"]) if "cfg_scale" in payload else 7
        payload["subseed"] = int(payload["subseed"]) if "subseed" in payload else -1
        payload["subseed_strength"] = float(payload["subseed_strength"]) if "subseed_strength" in payload else 0

        # Request txt2img endpoint using data
        responseTxt2img = requests.post(url=f'{STABLE_DIFFUSION_URL}/sdapi/v1/txt2img', json=payload)

        # Check if SD returned success
        if responseTxt2img.status_code != 200:
            print(f'    [-] ERROR: {responseTxt2img.json()["detail"]["msg"]}')
            continue

        # Write scaled file to disk
        open(f"./image_output/{curTime}/{inputImgName}", "wb+").write(base64.b64decode(responseTxt2img.json()["images"][0]))
        print(f"    [+] Finished upscaling {inputImgName}")
        imageId += 1
    except Exception as e:
        print(f"    [-] {traceback.format_exc()}")