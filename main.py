import pickle

import pygame
import random
import math
import neat
import os

pygame.init()

gen = 0
highscore = 0


class Goal:
    def __init__(self):
        self.i = random.randint(0, 10)
        self.j = math.radians(random.randint(0, 360))
        self.x = 50 + math.cos(self.j) * self.i
        self.y = 50 + math.sin(self.j) * self.i

    def reset(self):
        self.i = random.randint(0, 50)
        self.j = math.radians(random.randint(0, 360))
        self.x = 50 + math.cos(self.j) * self.i
        self.y = 50 + math.sin(self.j) * self.i

    def draw(self, window):
        pygame.draw.circle(window, (255, 0, 255), (self.x * 8, self.y * 8), 4)


class Terminator:
    def __init__(self):
        self.x = random.randint(0, 200)
        self.y = random.randint(0, 112)

    def draw(self, window):
        pygame.draw.circle(window, (200, 0, 200), (self.x * 4, self.y * 4), 6)

    def run(self, players):
        c = None
        d = 9999999999999999999999
        for player in players:
            e = math.dist((player.x, player.y), (self.x, self.y))
            if d >= e:
                d = e
                c = player

            if d < 2:
                return player

        direction = math.atan2(c.y - self.y, c.x - self.x)
        self.x += math.cos(direction) / 2.1
        self.y += math.sin(direction) / 2.1


class Player:
    def __init__(self):
        self.fitness = 0

        self.x = 100
        self.y = 100

        self.color = [random.randint(0, 255) for _ in range(3)]

        self.direction = math.radians(random.randint(0, 360))

        self.active = True

    def draw(self, window, path):
        pygame.draw.circle(path, self.color, (self.x * 4, self.y * 4), 1)
        pygame.draw.circle(window, self.color, (self.x * 4, self.y * 4), 4)

    def move(self, border_rad):
        self.x += math.cos(self.direction) / 2
        self.y += math.sin(self.direction) / 2

        w = pygame.Rect(0, 0, 800, 800)

        if math.dist((self.x * 4, self.y * 4), w.center) > border_rad:
            self.active = False
            self.fitness = 0


mode = "ALL"


def eval_genomes(genomes, config):
    global gen, highscore, mode

    window = pygame.Surface((800, 800))
    path = pygame.Surface((800, 800))
    display = pygame.display.set_mode(window.get_size())

    goals = [Goal() for _ in range(50)]

    terminator = Terminator()

    nets = []
    ge = []
    players = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Player())
        ge.append(genome)

    clock = pygame.time.Clock()

    path.fill((255, 255, 255))

    while len(players) > 0:
        clock.tick()

        window.blit(path, (0, 0))

        t = terminator.run(players)
        border_rad = 400
        pygame.draw.circle(window, (200, 0, 0), window.get_rect().center, border_rad, 2)

        if t is not None:
            o = players.index(t)
            del players[o]
            del nets[o]
            del ge[o]

        for i, player in enumerate(players):
            if player.active:
                ge[i].fitness = player.fitness
                player.move(border_rad)

                if mode == "ALL":
                    player.draw(window, path)

                d = 99999999999999999
                g = goals[0]

                for q, goal in enumerate(goals):
                    e = math.dist((player.x * 4, player.y * 4), (goal.x * 8, goal.y * 8))
                    if e <= d:
                        d = e
                        g = goal

                    if e <= 3:
                        goals[q].reset()
                        player.fitness += 1000

                player.fitness += (1 / e) * 10

                o = nets[i].activate([border_rad, math.dist((player.x * 4, player.y * 4), window.get_rect().center), math.atan2(player.y * 4 - g.y * 8, player.x * 4 - g.x * 8), math.atan2(player.y * 4 - terminator.y * 4, player.x * 4 - terminator.x * 4)])
                
                if o.index(max(o)) == 0:
                    player.direction += .2
                elif o.index(max(o)) == 1:
                    player.direction -= .2

                if player.fitness >= highscore:
                    highscore = player.fitness
                    pickle.dump(nets[i], open("best.pkl", "wb"))

            else:
                del players[i]
                del ge[i]
                del nets[i]

            if mode == "ONE" and i == 0:
                player.draw(window, path)

        for goal in goals:
            goal.draw(window)

        terminator.draw(window)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    if mode == "ALL":
                        mode = "ONE"
                    else:
                        mode = "ALL"

        pygame.display.set_caption("Generation: " + str(gen) + " FPS: " + str(int(clock.get_fps())) + " Players: " + str(len(players)))
        display.blit(window, (0, 0))
        pygame.display.update()

    pygame.image.save(window, "images/GEN" + str(gen) + ".jpg")
    gen += 1


def run(file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 5000)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
