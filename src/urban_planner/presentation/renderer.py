import logging
from typing import Optional, List

import pygame

from urban_planner.domain.geometries import Plot, Outline

logger = logging.getLogger(__name__)

class Renderer:
    """Handles all Pygame rendering for the UrbanPlannerEnv. A pure drawing utility."""

    THEME = {
        "background": (255, 255, 255),
        "plot_border": (100, 100, 100),
        "permitted_border": (125, 0, 0),
        "outline_border": (0, 0, 0),
        "outline_node": (0, 0, 0),
        "text_color": (0, 0, 0),
        "font_size": 24,
        "font_name": None,
        "line_height": 25
    }

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

    def init(self):
        """Initializes Pygame, the display window, and other resources."""
        if self.window is not None:
            return

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(self.THEME["font_name"], self.THEME["font_size"])
        logger.info("Renderer initialized.")

    def _render_info_text(self, canvas: pygame.Surface, info_lines: List[str]):
        """Renders lines of text in the top-left corner."""
        for i, text in enumerate(info_lines):
            text_surface = self.font.render(text, True, self.THEME["text_color"])
            canvas.blit(text_surface, (10, 10 + i * self.THEME["line_height"]))

    def render_frame(self, plot: 'Plot', outline: 'Outline', info_lines: List[str]):
        """
        Renders a single frame of the environment.
        Accepts geometric data and a pre-formatted list of strings to display.
        """
        assert self.window is not None, "Renderer must be initialized with .init() before rendering."

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.THEME["background"])

        pygame.draw.polygon(canvas, self.THEME["plot_border"], plot.coords, 2)
        pygame.draw.polygon(canvas, self.THEME["permitted_border"],
                            list(plot.permitted_outline.exterior.coords), 2)
        pygame.draw.polygon(canvas, self.THEME["outline_border"], outline.coords, 2)
        for point in outline.coords:
            pygame.draw.circle(canvas, self.THEME["outline_node"], point, 4)

        if info_lines:
            self._render_info_text(canvas, info_lines)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        """Closes the Pygame window and quits Pygame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            logger.info("Renderer closed.")