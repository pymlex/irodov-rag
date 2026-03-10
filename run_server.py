import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok

from src.server import app

token = os.environ.get("NGROK_AUTH_TOKEN")
if token:
    ngrok.set_auth_token(token)

tunnel = ngrok.connect(8000)

print("API URL:")
print(tunnel.public_url)

nest_asyncio.apply()

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000
)