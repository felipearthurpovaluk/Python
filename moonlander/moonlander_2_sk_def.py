# -*- coding: utf-8 -*-
"""
Recriação do jogo Sonda Lunar (Lunar Lander)

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import pygame
    
import sys
import traceback
import math
import random
import bisect
import pickle
import time
import copy
import neuralnetwork as nn
import numpy as np
import datetime
from sklearn.svm import LinearSVC

class GameConstants:
    #                  R    G    B
    ColorWhite     = (255, 255, 255)
    ColorBlack     = (  0,   0,   0)
    ColorRed       = (255,   0,   0)
    ColorGreen     = (  0, 255,   0)
    ColorDarkGreen = (  0, 155,   0)
    ColorDarkGray  = ( 40,  40,  40)
    BackgroundColor = ColorBlack
    
    screenScale = 1
    screenWidth = screenScale*800
    screenHeight = screenScale*600
    
    randomSeed = 0
    terrainR = 1.0
    
    landerWidth = 20
    landerHeight = 20
    
    FPS = 30
    
    fontSize = 20
    
    gravity = 1.622*5 # moon gravity is 1.622m/s2, 5 is just a scale factor for the game
    thrustAcceleration = 2.124*gravity # engine thrust-to-weight ratio
    rotationFilter = 0.75 # higher is more difficult, must be in range [0,1]
    topSpeed = 0.35*1000 # not actually implemented but it's a good idea
    drag = 0.0003 # this is a wild guess - just a tiiiny bit of aerodynamics

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        
    def as_tuple(self):
        return (self.x, self.y)

class Line:            
    def __init__(self, x0, y0, x1, y1, landingSpot):
        self.p0 = Point(x0, y0)
        self.p1 = Point(x1, y1)
        self.landingSpot = landingSpot
        
    def as_tuple(self):
        return (self.p0.as_tuple(), self.p1.as_tuple())
        
# Midpoint displacement algorithm for random terrain - better alternative to Perlin noise
# Source: https://bitesofcode.wordpress.com/2016/12/23/landscape-generation-using-midpoint-displacement/
# PS: this generates 2^num_of_iterations points
def midpoint_displacement(start, end, roughness, vertical_displacement=None,
                          num_of_iterations=16):
    """
    Given a straight line segment specified by a starting point and an endpoint
    in the form of [starting_point_x, starting_point_y] and [endpoint_x, endpoint_y],
    a roughness value > 0, an initial vertical displacement and a number of
    iterations > 0 applies the  midpoint algorithm to the specified segment and
    returns the obtained list of points in the form
    points = [[x_0, y_0],[x_1, y_1],...,[x_n, y_n]]
    """
    # Final number of points = (2^iterations)+1
    if vertical_displacement is None:
        # if no initial displacement is specified set displacement to:
        #  (y_start+y_end)/2
        vertical_displacement = (start[1]+end[1])/2
    # Data structure that stores the points is a list of lists where
    # each sublist represents a point and holds its x and y coordinates:
    # points=[[x_0, y_0],[x_1, y_1],...,[x_n, y_n]]
    #              |          |              |
    #           point 0    point 1        point n
    # The points list is always kept sorted from smallest to biggest x-value
    points = [start, end]
    iteration = 1
    while iteration <= num_of_iterations:
        # Since the list of points will be dynamically updated with the new computed
        # points after each midpoint displacement it is necessary to create a copy
        # of the state at the beginning of the iteration so we can iterate over
        # the original sequence.
        # Tuple type is used for security reasons since they are immutable in Python.
        points_tup = tuple(points)
        for i in range(len(points_tup)-1):
            # Calculate x and y midpoint coordinates:
            # [(x_i+x_(i+1))/2, (y_i+y_(i+1))/2]
            midpoint = list(map(lambda x: (points_tup[i][x]+points_tup[i+1][x])/2, [0, 1]))
            # Displace midpoint y-coordinate
            midpoint[1] += random.choice([-vertical_displacement, vertical_displacement])
            # Insert the displaced midpoint in the current list of points         
            bisect.insort(points, midpoint)
            # bisect allows to insert an element in a list so that its order
            # is preserved.
            # By default the maintained order is from smallest to biggest list first
            # element which is what we want.
        # Reduce displacement range
        vertical_displacement *= 2 ** (-roughness)
        # update number of iterations
        iteration += 1
    return points

class Game:
    class GameState:
        # Lander properties
        landerx = 0
        landery = 0
        landerdx = 0
        landerdy = 0
        landerddx = 0
        landerddy = 0
        landerRotation = 0
        landerLanded = False
        landerExploded = False

        # Lander implicit inputs (control inputs)
        landerThrust = 0             # saturated [0,1]
        landerTargetRotation = 0     # saturated [-90, 90]
        
        # Lander explicit inputs (player controlled inputs)
        rotateLanderLeft = False
        rotateLanderRight = False
        increaseLanderThrust = False

        # Metrics
        score = 0
        time = 0
        fuel = 0
    
    def __init__(self, width, height, ts, expectUserInputs=True):
        self.width = width
        self.height = height
        self.ts = ts # time step
        self.expectUserInputs = expectUserInputs
        
        # Game state list - stores a state for each time step
        gs = Game.GameState()
        gs.landerx = width - width//2
        gs.landery = height - height//6
        self.states = [gs]
        
        # Landscape, creates self.landscape and self.lines
        self.landscapeGenerator(width, height)
        
        # Determines if simulation is active or not
        self.alive = True

    def landscapeGenerator(self, width, height):    #felipe::define o mapa
        #lines = []
        #line = Line(0, height//3, width, height//3, True)
        #lines.append(line)
        #return lines
        
        
        # Initial points
        points = [[0, height//3], [width, height//3]]

        # Create points for landscape with midpoint displacement
        points = midpoint_displacement(points[0], points[1], GameConstants.terrainR, height//3, 5)
        
        # Map points to the base of our window (up to 98%)
        #points = list(map(lambda p: [p[0], min(height*0.98, height-p[1])], points))
        points = list(map(lambda p: [p[0], max(height*0.02, p[1])], points))
        
        # Sort at least 3 landing spots (there may be more if there are points below 98% window)
        landingSpotsCount = 1
        landingSpots = random.sample(points[:-1], landingSpotsCount)
            
        # Create lines for our landscape
        self.landscape = []
        self.lines = []
        
        last_point = points[0]
        for point in points[1:]:
            landingSpot = False
            
            # If it's a designated landing spot make it flat
            if last_point in landingSpots:
                point[1] = last_point[1]

            # If it's flat then it's a landing spot
            if point[1] == last_point[1]:
                landingSpot = True

            line = Line(last_point[0], last_point[1], point[0], point[1], landingSpot)            
            self.landscape.append(line)
            
            line = Line(last_point[0], self.height - last_point[1], point[0], self.height - point[1], landingSpot)            
            self.lines.append(line)
            
            last_point = point
    


    def checkCollision(self, gs): #felipe::verifica colisão da nave
        w = GameConstants.landerWidth
        h = GameConstants.landerHeight
        
        # Lander corners (vertices)
        landerLeft = gs.landerx - w/2
        landerRight = gs.landerx + w/2
        landerBottom = gs.landery - h/2
        landerTop = gs.landery + h/2
        
        v0 = (landerRight, landerTop)
        v1 = (landerRight, landerBottom)
        v2 = (landerLeft, landerTop)
        v3 = (landerLeft, landerBottom)
        v = [v0, v1, v2, v3]
        
        # Check all lines in landscape
        for l in self.landscape:
            # If the left-most of right-most vertices are in this line's bounds
            if l.p0.x <= landerLeft <= l.p1.x or l.p0.x <= landerRight <= l.p1.x:
                m = (l.p1.y - l.p0.y)/(l.p1.x - l.p0.x)
                y = lambda x: m*(x - l.p0.x) + l.p0.y

                # consider only vertices in domain (x) of this line
                inDomain = list(map(lambda vi: l.p0.x <= vi[0] <= l.p1.x, v))
                vInDomain =  [vi for (vi, b) in zip(v, inDomain) if b]
                
                # check if any vertices are under (above in pixels) the line
                if any(map(lambda vi: vi[1] <= y(vi[0]), vInDomain)):
                    '''
                    print('BOOM')
                    print('p0 {} p1 {}'.format(l.p0.as_tuple(),l.p1.as_tuple()))
                    print('v0 {} v1 {} v2 {} v3 {}'.format(v0,v1,v2,v3))
                    print(list(map(lambda vi: vi[1] <= y(vi[0]), v)))
                    print(list(map(lambda vi: vi[0], v)))
                    print(list(map(lambda vi: vi[1], v)))
                    print(list(map(lambda vi: y(vi[0]), v)))
                    '''
                    if l.landingSpot and abs(gs.landerRotation) <= 15 and abs(gs.landerdy) <= 20:
                        gs.landerLanded = True
                    else:
                        gs.landerExploded = True
    
    # Implements a game tick
    # Each call simulates a world step
    def update(self):
        # If the game is done, do nothing
        #if self.landerLanded or self.landerExploded:
        if not self.alive:
            return

        # Get current game state
        gs = self.states[-1]
        
        # Update time tick
        gs.time += self.ts
        
        # Process user inputs - in the absence of user inputs, implicit inputs are expected (gs.landerThrust and gs.landerTargetRotation)
        if self.expectUserInputs:
            if gs.rotateLanderLeft:
                gs.landerTargetRotation += 0.5
            elif gs.rotateLanderRight:
                gs.landerTargetRotation -= 0.5

            if gs.increaseLanderThrust:
                gs.landerThrust += 0.1
            else:
                gs.landerThrust -= 0.1
            
        # Saturate rotation and thrust
        if gs.landerTargetRotation > 90:
                gs.landerTargetRotation = 90
        elif gs.landerTargetRotation < -90:
                gs.landerTargetRotation = -90

        if gs.landerThrust > 1:
            gs.landerThrust = 1 #0.7 + 3*random.random()/10# gives a nice turbulence
        elif gs.landerThrust < 0:
            gs.landerThrust = 0

        # TODO: it might be interesting to limit implicit inputs to a few digits
        #gs.landerTargetRotation = round(gs.landerTargetRotation, 2)
        #gs.landerThrust = round(gs.landerThrust, 2)

        # lowpass filter on rotation to simulate rotation dynamics
        rf = GameConstants.rotationFilter
        gs.landerRotation = gs.landerTargetRotation*(1 - rf) + gs.landerRotation*rf            
            
        # Fuel consumption is proportional to integral of thrust
        gs.fuel += gs.landerThrust
        
        ## First order integration (Newton method) - moves the lander
        # PS: *IMPORTANT* The order of x/y dx/dy ddx/ddy changes the final result!
        # We want dx/dt = F(t, ...). As such, each ODE is dependent only on PAST terms. Past terms is badly implemented for landerddx/y, should be fixed (TODO).
        # x/y
        gs.landerx += gs.landerdx*self.ts
        gs.landery += gs.landerdy*self.ts
        
        # dx/dy considering drag
        gs.landerdx += gs.landerddx*self.ts
        gs.landerdy += gs.landerddy*self.ts
                                                  
        # Update acceleration coefficients based on thrust
        gs.landerddx = -gs.landerThrust*GameConstants.thrustAcceleration*math.sin(math.radians(gs.landerRotation)) - GameConstants.drag*gs.landerdx
        gs.landerddy = gs.landerThrust*GameConstants.thrustAcceleration*math.cos(math.radians(gs.landerRotation)) - GameConstants.gravity - GameConstants.drag*gs.landerdy

        # Go around the moon when near an edge        
        gs.landerx %= self.width
                                                  
        # Check for collisions                                                  
        self.checkCollision(gs)
        
        # Signal if the game ended
        if gs.landerLanded or gs.landerExploded:
                self.alive = False
       

def landscapeDraw(screen, game):
    rects = []

    rects += [pygame.draw.lines(screen, GameConstants.ColorWhite, False, [p for l in game.lines for p in l.as_tuple()], 2)]

    for l in game.lines:
        if l.landingSpot:
            rects += [pygame.draw.line(screen, GameConstants.ColorGreen, l.p0.as_tuple(), l.p1.as_tuple(), 3)]
    
    
    return rects

def infoDraw(screen, font, game):
    rects = []
    
    aa = False
    fontSize = GameConstants.fontSize
    xInfo1, yInfo1 = 10, 15
    
    gs = game.states[-1]
    
    fontSurface = font.render("SCORE: {:04d}".format(gs.score), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1]))
    
    fontSurface = font.render("TIME:  {:01d}:{:02d}".format(int(gs.time)//60, int(gs.time) % 60), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1+fontSize]))
    
    fontSurface = font.render("FUEL:  {:04d}".format(int(gs.fuel)), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1+2*fontSize]))
    
    
    fontSurface = font.render("ALTITUDE:          {:04d}".format(int(gs.landery)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("HORIZONTAL SPEED:  {:04d}".format(int(gs.landerdx)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("VERTICAL SPEED:    {:04d}".format(int(gs.landerdy)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + 2*fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("ROTATION:          {:04d}".format(int((gs.landerRotation))), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + 3*fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    return rects

def lunarLanderDraw(screen, game):
    rects = []
    
    w = GameConstants.landerWidth
    h = GameConstants.landerHeight
    gs = game.states[-1]
    
    surf = pygame.Surface((w+1, h+1))
    surf.set_colorkey((0,0,0)) # transparency
    
    #x = int(game.width - game.landerx)
    #y = int(game.height - game.landery)
    x = 0
    y = 0
    
    if gs.landerLanded:
        color = GameConstants.ColorGreen
    elif gs.landerExploded:
        color = GameConstants.ColorRed
    else:
        color = GameConstants.ColorWhite
    
    # Draw encompassing rectangle
    #pygame.draw.rect(surf, color, (x, y, w, h), 1)
    
    # Draw circle
    pygame.draw.circle(surf, color, (x+w//2, y+h//5), h//5, 1)
    
    # Draw small rectangle
    pygame.draw.rect(surf, color, (x+w//5, y+2*h//5, w-2*w//5, h//6), 1)
    
    # Draw legs
    pygame.draw.line(surf, color, (x+1.5*w//5, y+2*h//5+h//6), (x+0.5*w//5, y+h), 1)
    pygame.draw.line(surf, color, (x + w - 1.5*w//5, y+2*h//5+h//6), (x + w - 0.5*w//5, y+h), 1)
    
    # Draw feet
    pygame.draw.line(surf, color, (x, y+h), (x+1.0*w//5, y+h), 2)
    pygame.draw.line(surf, color, (x + w, y+h), (x + w - 1.0*w//5, y+h), 2)
    
    # Draw thrust
    tby = y+2*h//5+h//6 # thrust base y
    tm = y + h - tby    # thrust multiplier for height
    pygame.draw.line(surf, color, (x+3*w//5, tby), (x+w//2, tby+tm*gs.landerThrust), 1)
    pygame.draw.line(surf, color, (x + w - 3*w//5, tby), (x + w//2, tby+tm*gs.landerThrust), 1)
    
    # Apply rotation and center
    rotSurf = pygame.transform.rotate(surf, gs.landerRotation)
    r = pygame.Rect(rotSurf.get_rect())
    r.center = (gs.landerx, game.height - gs.landery)
    rects += [screen.blit(rotSurf, r)]
    
    return rects

def draw(screen, font, game):
    rects = []
            
    rects += [screen.fill(GameConstants.BackgroundColor)]
    rects += landscapeDraw(screen, game)
    rects += infoDraw(screen, font, game)
    rects += lunarLanderDraw(screen, game)
    
    return rects

def initialize():
    random.seed(GameConstants.randomSeed)
    pygame.init()
    game = Game(int(GameConstants.screenWidth), int(GameConstants.screenHeight), 1/GameConstants.FPS)
    font = pygame.font.SysFont('Courier', GameConstants.fontSize)
    fpsClock = pygame.time.Clock()

    # Create display surface
    screen = pygame.display.set_mode((game.width, game.height), pygame.DOUBLEBUF)
    screen.fill((0, 0, 0))
        
    return screen, font, game, fpsClock

def handleEvents(game):
    gs = game.states[-1]
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                gs.rotateLanderLeft = True
            elif event.key == pygame.K_RIGHT:
                gs.rotateLanderRight = True
            elif event.key == pygame.K_UP:
                gs.increaseLanderThrust = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                gs.rotateLanderLeft = False
            elif event.key == pygame.K_RIGHT:
                gs.rotateLanderRight = False
            elif event.key == pygame.K_UP:
                gs.increaseLanderThrust = False
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

def defaultTrain():
    # x = Obstruído (10 un.)/ Velocidade > 20/	Velocidade < -10/	Distancia ao chão/	Spot na esquerda/	No SPOT/	Inclinacao > 0	/Inclinação = 0
    # y = 	UP/	RIGTH/	LEFT
    
    x = np.array([[1,0,1,1,1,0,1,0,0,0],
[1,0,1,1,0,1,0,1,0,0],
[1,0,1,0,1,0,0,0,0,0],
[1,0,1,0,0,1,1,0,0,0],
[1,0,0,1,1,0,0,1,0,0],
[1,0,0,1,0,1,0,0,0,0],
[1,0,0,0,0,0,0,0,0,0],
[1,0,0,0,0,1,0,1,0,0],
[0,0,1,1,0,0,0,0,0,0],
[0,0,1,1,0,1,1,0,0,0],
[0,0,1,0,0,0,0,1,0,0],
[0,0,1,0,0,1,0,0,0,0],
[0,0,0,1,1,0,1,0,0,0],
[0,0,0,1,0,1,0,1,0,0],
[0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,1,0,1,0,0],
[1,1,0,0,0,1,0,0,0,0],
[1,1,0,1,1,0,0,1,0,0],
[0,1,0,0,1,0,1,0,0,0],
[0,1,0,1,1,0,0,0,0,0],
[1,1,0,1,1,0,0,0,0,0],
[1,0,1,1,1,0,1,0,1,0],
[1,0,1,1,0,1,1,0,1,0],
[1,0,1,0,1,0,1,0,1,0],
[1,0,1,0,0,1,1,0,1,0],
[1,0,0,1,1,0,1,0,1,0],
[1,0,0,1,0,1,0,0,0,1],
[1,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,1,0,0,0,1],
[0,0,1,1,0,0,0,0,0,1],
[0,0,1,1,0,1,0,0,0,1],
[0,0,1,1,1,0,1,0,1,0],
[0,0,1,1,0,1,1,0,1,0],
[0,0,0,0,1,0,1,0,1,0],
[0,1,0,0,1,1,1,0,1,0],
[0,0,0,0,1,1,1,0,0,0]])

    ''',
    [0,1,0,1,1,1,1,0,1,0],
    [0,0,0,1,1,1,1,0,1,0],
    [0,1,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0]'''

    y = np.array([[1,0,1],
[1,0,0],
[1,0,1],
[1,1,0],
[1,0,1],
[1,0,0],
[1,1,0],
[1,1,0],
[1,1,0],
[1,1,0],
[0,1,0],
[1,0,0],
[1,0,0],
[0,0,0],
[1,0,1],
[0,0,0],
[1,1,0],
[1,0,1],
[0,0,1],
[1,0,1],
[0,0,1],
[1,1,0],
[0,1,0],
[1,1,0],
[0,1,0],
[0,1,0],
[1,0,1],
[1,0,1],
[0,0,1],
[1,1,0],
[1,0,1],
[1,1,0],
[1,1,0],
[0,1,0],
[0,1,0],
[0,1,0]])

    ''',
    [1,1,0],
    [1,1,0],
    [1,1,0],
    [1,1,0]'''

    yUP = y[:, 0]
    yRIGTH = y[:, 1]
    yLEFT = y[:, 2]

    xUP = x[:, [0,1,2]]
    xSIZES = x[:, [3,4,5,6,7,8,9]]

    rUP = LinearSVC()
    rUP.fit(xUP,yUP)

    rRIGTH = LinearSVC()
    rRIGTH.fit(xSIZES,yRIGTH)

    rLEFT = LinearSVC()
    rLEFT.fit(xSIZES,yLEFT)

    return rUP, rRIGTH, rLEFT

def auto(game, rUP, rRIGTH, rLEFT): #felipe:: bot
        
    ''' Preciso das informações: 
            Obstruído (10 un.)	    ok (50%)
            Velocidade < -10	    
            Distancia ao chão	
            Spot na esquerda	
            No SPOT
            Inclinacao > 0	
            Inclinação = 0
    '''
    wi = GameConstants.landerWidth
    he = GameConstants.landerHeight
    
    ## verificando se está obstruído
    landerLeft = game.states[-1].landerx - wi
    landerRight = game.states[-1].landerx  + wi
    landerBottom = game.states[-1].landery - he/2

    # :: Obstruído (10 un.)/ Velocidade > 20/	Velocidade < -10/	Distancia ao chão/	No SPOT/ Spot na esquerda/	Inclinacao > 0	/Inclinação = 0/ Inclinacao > 8/ Inclinacao < -8
    # Sempre começo com valores nulos e obtenho dos sensores
    DADOS = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 

    # :: OBTENDO O PONTO MÉDIO DE SPOT DE POUSO
    landingSpot_x = []
    for l in game.landscape:
        if l.landingSpot:
            landingSpot_x.append(l.p0.x)
            landingSpot_x.append(l.p1.x)

    midleLandingSpot = (landingSpot_x[0] + landingSpot_x[-1]) / 2

    #estado atual do cenário
    gs = game.states[-1]

    ########## SENSORES = VALOR DE ENTRADA DA REDE ########
    for l in game.landscape:
        ### :: Obstruído (10 un.) 
        if l.p0.y >= (gs.landery - he - 10): #10 unidades de distancia por garantia (obstrucao a (esquerda))
            #print(str(l.p0.y)+' >= '+str((gs.landery - he)))
            if (l.p0.x < landerLeft and midleLandingSpot < l.p0.x ) or (l.p0.x > landerRight and midleLandingSpot > l.p0.x ): #se spot está pra esquerda
                DADOS[0] = 1

        ### :: Distancia ao chão 
        ## se a diferença da posição atual da nave com o chao é menor que a base da nave, então está nessa verificação de chão e,
        # se está a menos de 30 posições do chão
        if abs(l.p0.x - gs.landerx) < wi and (landerBottom - l.p0.y) < 30: 
            DADOS[3] = 1
    
    ### :: Velocidade > 20
    #print(gs.landerdy)
    if gs.landerdy > 9:
        DADOS[1] = 1

    ### :: Velocidade < -10
    #print(game.states[-1].landerdy)
    if gs.landerdy < -7:
        DADOS[2] = 1

    ### Definições para girar
    ''' Spot na esquerda	
        No SPOT
        Inclinacao > 0	
        Inclinação = 0'''
    
    ## spot na esquerda
    if midleLandingSpot < landerLeft:
        DADOS[4] = 1

    ## No Spot (com 5 casas de verificação)
    print(str(midleLandingSpot) +' / '+ str(gs.landerx))
    if abs(midleLandingSpot - gs.landerx) <= 100:
        print(abs(midleLandingSpot - gs.landerx))
        DADOS[5] = 1

    
    ## auexplicativo
    if gs.landerRotation > 0:
        DADOS[6] = 1

    if gs.landerRotation == 0:
        DADOS[7] = 1

    if gs.landerRotation > 3:
        DADOS[8] = 1

    if gs.landerRotation < -3:
        DADOS[9] = 1

    #print(DADOS)

    
    #print(nn.networkCalculate(obstruido, layers))

    #result = nn.networkCalculate(DADOS, layers)
    #DADOS = [DADOS] #:: Tem que ser 2D, fazer oq

    DADOSUP = [[DADOS[0], DADOS[1], DADOS[2]]] #:: Tem que ser 2D, fazer oq
    DADOSSIZE = [[DADOS[3], DADOS[4], DADOS[5], DADOS[6], DADOS[7], DADOS[8] ,DADOS[9]]] 

    result = [rUP.predict(DADOSUP), rRIGTH.predict(DADOSSIZE), rLEFT.predict(DADOSSIZE)]

    if result[0] == 1:
        gs.increaseLanderThrust = True
    else:
        gs.increaseLanderThrust = False

    if result[1] == 1:
        gs.rotateLanderRight = True
    else:
        gs.rotateLanderRight = False

    if result[2] == 1:
        gs.rotateLanderLeft = True
    else:
        gs.rotateLanderLeft = False
    


def saveGame(game):
    with open("moonlander-{}.pickle".format(time.strftime("%Y%m%d-%H%M%S")), "wb") as f:
        pickle.dump([GameConstants, game], f)

        
def loadGame(fileName):
    with open(fileName, "rb") as f:
        GameConstants, game = pickle.load(f)
        
    return game
    
            
def mainGamePlayer():
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
            
        dt1 = datetime.datetime.now()

        rUP, rRIGTH, rLEFT = defaultTrain()

        dt2 = datetime.datetime.now()
        
        print('Tempo de treino: '+str(dt2 - dt1))

        # Main game loop
        while game.alive:
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events
            #handleEvents(game) #felipe:: esse método será alterado para o bot

            auto(game, rUP, rRIGTH, rLEFT)

            # Update world
            game.update()
            
            # Draw this world frame
            rects = draw(screen, font, game)
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)

        # save this playthrough    
        saveGame(game)
    except SystemExit:
        pass
    except Exception as e:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        #raise Exception from e
    finally:
        # close up shop
        pygame.quit() 
    
def mainGameAutonomous(thrust, rotation):
    try:
        # Verify if both thrust and rotation are the same size
        if(len(rotation) != len(thrust)):
            raise Exception('Thrust and rotation vectors must be the same size')
        
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()

        # Disable user inputs and supply implicit inputs (thrust and target rotation)
        game.expectUserInputs = False
              
        # Main game loop
        for t, r in zip(thrust, rotation):
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events from data
            game.states[-1].landerThrust = t
            game.states[-1].landerTargetRotation = r
                    
            # Update world
            game.update() 

            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)           

    except SystemExit:
        pass
    except:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
    finally:
        # close up shop
        pygame.quit() 

def mainGameAutonomousUserInputs(gamestates):
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        for state in gamestates:
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events from data
            game.states[-1].rotateLanderLeft = state.rotateLanderLeft
            game.states[-1].rotateLanderRight = state.rotateLanderRight
            game.states[-1].increaseLanderThrust = state.increaseLanderThrust
                    
            # Update world
            game.update() 

            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)           
              
    except SystemExit:
        pass
    except:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
    finally:
        # close up shop
        pygame.quit() 
    
if __name__ == "__main__":
    mainGamePlayer()

    # Load a save game
    #game = loadGame('moonlander-20180516-135318.pickle')
    
    # game.states[1:] is needed to skip initial state that already exists in Game
    # assuming initial conditions are the same in data and in simulation

    '''
    # Implicit inputs
    thrusts = [gs.landerThrust for gs in game.states[1:]]
    rotations = [gs.landerTargetRotation for gs in game.states[1:]]
    mainGameAutonomous(thrusts, rotations)
    
    # Explicit inputs from user
    mainGameAutonomousUserInputs(game.states[1:])
    '''