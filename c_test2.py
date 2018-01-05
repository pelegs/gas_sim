import numpy as np
import pygame
import vecfuncs

class line_segment:
    def __init__(self,
                 start,
                 end,
                 color):
        self.start = start.flatten()
        self.end   = end.flatten()
        self.color = color

    def draw(self, surface):
        pygame.draw.line(surface,
                         self.color,
                         self.start.astype(int),
                         self.end.astype(int),
                         3)

line1 = line_segment(start = np.random.uniform(0, 800, size=(1, 2)),
                     end   = np.random.uniform(0, 800, size=(1, 2)),
                     color = [255, 0, 0])

line2 = line_segment(start = np.random.uniform(0, 800, size=(1, 2)),
                     end   = np.random.uniform(0, 800, size=(1, 2)),
                     color = [0, 0, 255])

scr_size = 800
pygame.init()
screen = pygame.display.set_mode ((scr_size, scr_size))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                exit()
   
    screen.fill([255, 255, 255])
    line2.end = np.array(pygame.mouse.get_pos()).astype(np.float64)
    line1.draw(screen)
    line2.draw(screen)
    if vecfuncs.intersect(line1, line2): 
        c = vecfuncs.intersection_point(line1, line2)
        ref_size = np.linalg.norm(line2.end - c)
        ref = vecfuncs.reflect(line2.end - line2.start,
                               line1.end - line1.start)
        ref = vecfuncs.set_vec_mag(ref, ref_size)
        pygame.draw.line(screen,
                         [0, 255, 0],
                         c.astype(int),
                         (c + ref).astype(int),
                         3)
        pygame.draw.circle(screen,
                           [0, 0, 0],
                           c.astype(int),
                           5)
    pygame.display.update()
    #pygame.time.delay(500)
