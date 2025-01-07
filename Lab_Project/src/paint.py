import pygame
import sys

pygame.init()

BACKGROUND_COLOR = (0, 0, 0)
BRUSH_COLOR = (255, 255, 255)

window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Click to Paint")

running = True
mouse_pressed = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pressed = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_pressed = False

    if mouse_pressed:
        pos = pygame.mouse.get_pos()
        pygame.draw.circle(window, BRUSH_COLOR, pos, 5)

    pygame.display.flip()

pygame.quit()
sys.exit()

