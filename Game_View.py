import pygame
import numpy as np
import base64
import io
from PIL import Image
import threading
from API_Handler import StableDiffusionAPI, GenerationRequest


class SDGameView:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption("Dream")
        self.clock = pygame.time.Clock()
        self.running = False
        self.paused = False
        self.current_image = None
        self.camera_state = {"x": 0, "y": 0, "zoom": 1.0}
        self.prompt = "a beautiful landscape"
        self.generated_image_size = (512, 512)
        self.generated_image_init = None

        # Initialize API handler
        self.sd_api = StableDiffusionAPI()

    def _image_callback(self, image_data):
        """Handle completed generation requests"""
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        self.current_image = pygame.image.fromstring(
            image.tobytes(),
            image.size,
            image.mode
        )

    def start_generation(self):
        """Submit a new generation request"""
        request = GenerationRequest(
            prompt=self.prompt,
            callback=self._image_callback
        )
        self.sd_api.submit_request(request)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.toggle_pause()
                elif event.key == pygame.K_SPACE:
                    self.start_generation()
                elif event.key == pygame.K_q:
                    if self.paused:
                        self.running = False

        if not self.paused:
            # Get mouse movement
            rel_x, rel_y = pygame.mouse.get_rel()
            self.camera_state["x"] += rel_x * 0.1
            self.camera_state["y"] += rel_y * 0.1

            # Keep mouse centered
            pygame.mouse.set_pos(self.screen.get_width() // 2,
                                 self.screen.get_height() // 2)

    def render(self):
        self.screen.fill((0, 0, 0))

        if self.current_image:
            # Apply camera transformations
            img_rect = self.current_image.get_rect()
            img_rect.center = (
                self.screen.get_width() // 2 + self.camera_state["x"],
                self.screen.get_height() // 2 + self.camera_state["y"]
            )
            scaled_img = pygame.transform.scale(
                self.current_image,
                (int(img_rect.width * self.camera_state["zoom"]),
                 int(img_rect.height * self.camera_state["zoom"])))
            for x_offset in range(-1, 2):
                for y_offset in range(-1, 2):
                    transformed_img_rect = img_rect.copy()
                    transformed_img_rect.center = (
                        transformed_img_rect.center[0] + scaled_img.get_width() * x_offset,
                        transformed_img_rect.center[1] + scaled_img.get_height() * y_offset
                    )
                    # Flip the image according to where it is on screen
                    self.screen.blit(pygame.transform.flip(scaled_img, x_offset % 2 != 0, y_offset % 2 != 0), transformed_img_rect)

            pygame.display.flip()

    def toggle_pause(self):
        self.paused = not self.paused
        self.toggle_camera_lock(not self.paused)

    def toggle_camera_lock(self, lock_state = True):
        pygame.mouse.set_visible(not lock_state)
        pygame.event.set_grab(lock_state)

    def run(self):
        self.running = True
        self.toggle_camera_lock()

        # Start initial generation
        self.start_generation()

        while self.running:
            self.handle_input()
            if not self.paused:
                self.render()
            self.clock.tick(60)

        pygame.quit()