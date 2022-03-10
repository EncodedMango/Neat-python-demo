import pygame
import random
import math
import neat
import os

pygame.init()

gen = 0


class Goal:
    def __init__(self):
        self.x = random.randint(0, 100)
        self.y = random.randint(0, 56)

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
        self.x += math.cos(direction) / 2
        self.y += math.sin(direction) / 2


class Player:
    def __init__(self):
        self.fitness = 0

        self.x = 100
        self.y = 56

        self.color = [random.randint(0, 255) for _ in range(3)]

        self.direction = math.radians(random.randint(0, 360))

        self.active = True

    def draw(self, window, path):
        pygame.draw.rect(path, self.color, (self.x * 4 + 2, self.y * 4 + 2, 1, 1))
        pygame.draw.circle(window, self.color, (self.x * 4, self.y * 4), 4)

    def move(self):
        self.x += math.cos(self.direction) / 2
        self.y += math.sin(self.direction) / 2

        if not pygame.Rect(0, 0, 800, 450).collidepoint(self.x * 4, self.y * 4):
            self.active = False
            self.fitness = 0


def eval_genomes(genomes, config):
    global gen

    window = pygame.Surface((800, 450))
    path = pygame.Surface((800, 450))
    display = pygame.display.set_mode(window.get_size())

    terminator = Terminator()
    goals = [Goal() for _ in range(10)]

    nets = []
    ge = []
    players = []

    structs = []

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

        if t is not None:
            del ge[players.index(t)]
            del nets[players.index(t)]
            del players[players.index(t)]

        for i, player in enumerate(players):
            if player.active:
                ge[i].fitness = player.fitness
                player.move()
                player.draw(window, path)

                d = 99999999999999999
                g = goals[0]

                for q, goal in enumerate(goals):
                    e = math.dist((player.x, player.y), (goal.x, goal.y))
                    if e <= d:
                        d = e
                        g = goal

                    if e <= 10:
                        goals[q].x = random.randint(0, 100)
                        goals[q].y = random.randint(0, 56)
                        player.fitness += 1000
                        
                player.fitness += 1 / e

                o = nets[i].activate([math.atan2(player.y - terminator.y, player.x - terminator.x), math.atan2(player.y - g.y, player.x - g.x), 800 - player.x, 450 - player.y])
                
                if o.index(max(o)) == 0:
                    player.direction += 0.1
                elif o.index(max(o)) == 1:
                    player.direction -= 0.1

            else:
                del players[i]
                del ge[i]
                del nets[i]

        terminator.draw(window)

        for goal in goals:
            goal.draw(window)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        pygame.display.set_caption("Generation: " + str(gen) + " FPS: " + str(int(clock.get_fps())) + " Players: " + str(len(players)))
        display.blit(window, (0, 0))
        pygame.display.update()

    gen += 1


def run(file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 500000000000000000000000000000000000000000000)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
