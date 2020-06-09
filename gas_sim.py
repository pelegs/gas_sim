import numpy as np
import pygame
import sys
from itertools import chain
import time
import argparse
import vecfuncs
import gaslibc

# colors
colors = {
          'white': [255, 255, 255],
          'black': [0, 0, 0],
          'red'  : [255, 0, 0],
          'green': [0, 255, 0],
          'blue' : [0, 0, 255],
         }

directions = {
              'horiz': np.array([1, 0]),
              'vert' : np.array([0, 1]),
             }              

class ball:
    cell = np.zeros(2)

    def __init__(self,
                 pos   = np.zeros(2),
                 vel   = np.zeros(2),
                 rad   = 1.0,
                 mass  = 1.0,
                 color = colors['black'],
                 ID    = -1,
                 grid  = None):

        self.pos    = pos.astype(np.float64).flatten()
        self.vel    = vel.astype(np.float64).flatten()
        self.rad    = rad
        self.mass   = mass 
        self.color  = color
        self.events = []
        self.ID     = ID
        if grid is not None:
            self.cell = np.array([int(np.floor(x*grid.N)) for x in self.pos])
        self.neighbors = []
        self.MSD = 0.0

    def update_neighbors(self, grid):
        cells = list(chain(*[grid.cells[i][j]
                     for (i, j) in vecfuncs.neighbor_indices(self.cell[0], self.cell[1], grid.N)]))
        self.neighbors = [b for b in cells if b is not self]

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), int(self.rad))

    def move(self, dt, gravity=False, g=np.array([0, 1])):
        prev_pos = self.pos
        if gravity:
            self.vel = self.vel + g * dt
        self.pos = self.pos + self.vel * dt
        self.cell = np.array([int(np.floor(x*grid.N))+1 for x in self.pos])
        self.MSD += np.linalg.norm(self.pos - prev_pos)**2

    def bounce(self, wall):
        self.vel = vecfuncs.reflect(self.vel, wall.direction)

    def Ek(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

class wall:
    def __init__(self,
                 direction = directions['horiz'],
                 start     = np.zeros(2),
                 length    = 1,
                 width     = 5,
                 color     = colors['black'],
                 ID        = -1):

        self.direction = direction.astype(np.float64).flatten()
        self.start     = start.astype(np.float64).flatten()
        self.length    = length
        self.end       = start + length * direction
        self.width     = width
        self.color     = color
        self.ID        = ID

    def draw(self, surface): 
        pygame.draw.line(surface, self.color, self.start.astype(int), self.end.astype(int), self.width)


class sim_grid:
    def __init__(self, N):
        self.cells = [[[] for _ in range(N)]
                          for _ in range(N)]
        self.N = N

    def insert(self, b, L):
        cell = np.array([int(np.floor(x/L*self.N)) for x in b.pos])
        b.cell = cell
        if (0 <= cell[0] < self.N) and (0 <= cell[1] < self.N):
            self.cells[cell[0]][cell[1]].append(b)

    def reset(self):
        self.cells = [[[] for _ in range(self.N)]
                          for _ in range(self.N)]
       
    def draw(self, surface, L):
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                if cell == []:
                    color = 3*[200]
                    pygame.draw.rect(surface, color, [i*L/self.N, j*L/self.N, L/self.N, L/self.N], 2)
                else:
                    for c in vecfuncs.neighbor_indices(i, j, self.N):
                        color = [100, 0, 0]
                        pygame.draw.rect(surface, color, [c[0]*L/self.N, c[1]*L/self.N, L/self.N, L/self.N], 2)

def zero_cm_vel(balls):
    cm_vel = np.sum(b.vel for b in balls)
    balls[0].vel = balls[0].vel - cm_vel

def place_particles(num_particles,
                    pos_min, pos_max,
                    vel_sigma,
                    rad, mass,
                    color):
    particles = []
    i = 0
    while i < num_particles:
        c = np.random.uniform(pos_min, pos_max, size=(1,2)).astype(np.float64).flatten()
        overlap = False
        for b in particles:
            if vecfuncs.distance(c, b.pos) <= 2*rad:
                overlap = True
                break
        if not overlap:
            v = np.random.normal(0, vel_sigma, size=(1,2))
            particles.append(ball(pos  = c,
                                  vel  = v,
                                  rad  = rad,
                                  mass = mass,
                                  ID   = i,
                                  color = color))
            i += 1
    print(i)
    return particles

parser = argparse.ArgumentParser(description="A simple 2D gas simulation")
parser.add_argument('-i','--input',  help='Input file', required=False)
parser.add_argument('-p','--place',  type=bool,  required=False, default=False)
parser.add_argument('-r','--radius', type=float, required=False, default=10.0)
parser.add_argument('-s','--speed',  type=float, required=False, default=50.0)
parser.add_argument('-c','--screen', type=int,   required=False, default=800)
parser.add_argument('-t','--dt',     type=float, required=False, default=0.1)
parser.add_argument('-n','--num',    type=int,   required=False, default=100)
parser.add_argument('-g','--grav',   type=float, required=False, default=3.5)
parser.parse_args()
args = vars(parser.parse_args())

''' !! should be moved to a function !! '''
if args['place']:
    rad = args['radius']
    scr_size = args['screen']
    dt = args['dt']
    N_balls = args['num']
    balls = place_particles(num_particles = N_balls,
                            pos_min = 10,
                            pos_max = 790,
                            vel_sigma = args['speed'],
                            rad = rad,
                            mass = 1,
                            color = [0, 0, 255])

    wall0 = wall(start     = np.array([0, 0]),
                 direction = np.array([1, 0]),
                 length    = 800.0,
                 width     = 2,
                 color     = [0, 0, 0])
    
    wall1 = wall(start     = np.array([0, 0]),
                 direction = np.array([0, 1]),
                 length    = 800.0,
                 width     = 2,
                 color     = [0, 0, 0])
    
    wall2 = wall(start     = np.array([800, 800]),
                 direction = np.array([-1, 0]),
                 length    = 800.0,
                 width     = 2,
                 color     = [0, 0, 0])
                
    wall3 = wall(start     = np.array([800, 800]),
                 direction = np.array([0, -1]),
                 length    = 800.0,
                 width     = 2,
                 color     = [0, 0, 0])

    walls = [wall0, wall1, wall2, wall3]
else:
    with open(args['input_file']) as f:
        lines = f.readlines()
    N_balls = int(lines[0][:-1].split()[0])
    rad = 5
    balls = []
    walls = []
    for i in range(1, N_balls+1):
        line = lines[i][:-1].split(' ')
        x, y, vx, vy, rad, mass = line[:6]
        color = np.array(line[6:9]).astype(int).flatten()
        balls.append(ball(pos   = np.array([float(x), float(y)]),
                          vel   = np.array([float(vx), float(vy)]),
                          rad   = int(rad),
                          mass  = float(mass),
                          color = color))
    zero_cm_vel(balls)
    N_walls = int(lines[N_balls+1][:-1].split()[0])
    for j in range(N_balls+2, N_balls+2+N_walls):
        line = lines[j][:-1].split(' ')
        sx, sy, ax, ay, L, w = line[:6]
        color = np.array(line[6:9]).astype(int).flatten()
        walls.append(wall(start     = np.array([float(sx), float(sy)]),
                          direction = np.array([float(ax), float(ay)]),
                          length    = float(L),
                          width     = int(w),
                          color     = color))
    last_line = lines[-1][:-1].split(' ')
    scr_size = int(last_line[0])
    dt = float(last_line[1])

num_grid_cells = int(scr_size / (int(rad) * 2))
grid = sim_grid(num_grid_cells)

MSD = np.zeros(N_balls)
t = 0

pygame.init()
screen = pygame.display.set_mode ((scr_size, scr_size))
while True:
    start_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_v:
                for ball in balls:
                    print(np.linalg.norm(ball.vel))

    screen.fill(colors['white'])
    
    # Insert balls into grid
    grid.reset()
    for ball in balls:
        grid.insert(ball, scr_size) 
    
    #grid.draw(screen, scr_size)
    for i, ball1 in enumerate(balls):
        ball1.update_neighbors(grid)
        for ball2 in ball1.neighbors:
            if gaslibc.time_ball_collision(ball1, ball2) < dt:
                gaslibc.ball_collision(ball1, ball2)
        for wall in walls:
            if gaslibc.time_wall_collision(ball1, wall) < dt:
                ball1.bounce(wall)
        ball1.move(dt, gravity=True, g=args['grav']*np.array([0, 1]))
        ball1.draw(screen)
        MSD[i] = ball.MSD
    for wall in walls:
        wall.draw(screen)
    pygame.display.update()
    #pygame.time.delay(500)

    #Ek_tot = sum([ball.Ek() for ball in balls ])
    #print('\rEk total =', Ek_tot, '              ', end='')
    #print(t, Ek_tot)
    
    #print(t, np.average(MSD))
    t += 1
    
    elapsed_time = time.time() - start_time
    print('\rFPS:', int(1/elapsed_time), '           ', end='')
