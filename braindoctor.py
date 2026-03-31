# Step1 : Setup Groq API key
import os
from groq import Groq
import base64

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Step2 : Convert image to required format
def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

# Step3 : Analyze image
def analyze_image_with_query(question, model,encoded_img):

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{encoded_img}"
                        }
                    }
                ]
            }
        ]
    )

    return(response.choices[0].message.content)