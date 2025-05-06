import aiohttp
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
import threading

@dataclass
class GenerationRequest:
    prompt: str
    seed: int = -1
    steps: int = 20
    width: int = 512
    height: int = 512
    callback: Optional[Callable] = None
    init_image: Optional[str] = None

class StableDiffusionAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url
        self.session = None
        self.queue = asyncio.Queue()
        self.current_request = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    async def _process_queue(self):
        while True:
            self.current_request = await self.queue.get()
            try:
                async with aiohttp.ClientSession() as session:
                    if self.current_request.init_image is None:
                        print('bello')
                        url = f"{self.base_url}/sdapi/v1/txt2img"
                        payload = {
                            "prompt": self.current_request.prompt,
                            "seed": self.current_request.seed,
                            "steps": 25,
                            "width": self.current_request.width,
                            "height": self.current_request.height
                        }
                    else:
                        print('Hello')
                        url = f"{self.base_url}/sdapi/v1/img2img"
                        payload = {
                            "init_images": [self.current_request.init_image],
                            "prompt": self.current_request.prompt,
                            "seed": self.current_request.seed,
                            "steps": self.current_request.steps,
                            "width": self.current_request.width,
                            "height": self.current_request.height,
                            "denoising_strength": 0.4,
                            "cfg_scale": 3,
                        }
                    response = await session.post(url, json=payload)
                    result = await response.json()
                    if self.current_request.callback:
                        self.current_request.callback(result['images'][0])
            except Exception as e:
                print(f"Generation failed: {e}")
            finally:
                self.current_request = None
                self.queue.task_done()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._process_queue())

    def submit_request(self, request: GenerationRequest):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, request)

    def cancel_current(self):
        if self.current_request:
            # TODO
            pass