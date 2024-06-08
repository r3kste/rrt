#!/usr/bin/python3

import copy
import math
import os
import random
import time
from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, pos: tuple[float, float], parent: Union["Node", None] = None):
        self.pos = pos
        self.x, self.y = pos
        self.parent = parent

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"

    def distance(self, other: Union["Node", None]) -> float:
        if other is None:
            return 0
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Obstacle:
    def __init__(self, corner1: Node, corner2: Node, is_wall: bool = False):
        self.corner1 = Node((min(corner1.x, corner2.x), min(corner1.y, corner2.y)))
        self.corner2 = Node((max(corner1.x, corner2.x), max(corner1.y, corner2.y)))
        self.width = self.corner2.x - self.corner1.x
        self.height = self.corner2.y - self.corner1.y
        self.is_wall = is_wall

    def __contains__(self, node: Node) -> bool:
        x1, y1 = self.corner1.x, self.corner1.y
        x2, y2 = self.corner2.x, self.corner2.y

        return x1 <= node.x <= x2 and y1 <= node.y <= y2

    def distance(self, node: Node) -> float:
        xc, yc = (self.corner1.x + self.corner2.x) / 2, (
            self.corner1.y + self.corner2.y
        ) / 2

        r = 0
        x1, y1 = node.x, node.y
        x1, y1 = x1 - xc, y1 - yc
        xl, xr = self.corner1.x, self.corner2.x
        yb, yt = self.corner1.y, self.corner2.y
        xl, xr = xl - xc, xr - xc
        yb, yt = yb - yc, yt - yc

        if y1 > yt:
            if x1 > xr:
                r = math.sqrt((x1 - xr) ** 2 + (y1 - yt) ** 2)
            elif x1 < xl:
                r = math.sqrt((x1 - xl) ** 2 + (y1 - yt) ** 2)
            else:
                r = yt - y1
        elif y1 < yb:
            if x1 > xr:
                r = math.sqrt((x1 - xr) ** 2 + (y1 - yb) ** 2)
            elif x1 < xl:
                r = math.sqrt((x1 - xl) ** 2 + (y1 - yb) ** 2)
            else:
                r = y1 - yb
        else:
            if x1 > xr:
                r = xr - x1
            elif x1 < xl:
                r = x1 - xl

        return abs(r)


class Environment:
    def __init__(
        self,
        dimensions: tuple[float, float],
        start: Node,
        goal: Node,
        obstacles: Union[list[Obstacle], None] = None,
        goal_radius: float = 1.0,
    ):
        if obstacles is None:
            obstacles = []
        self.width, self.height = dimensions
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.obstacles = obstacles

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def random_node(self, goal_bias: float = 0.2) -> Node:
        if random.random() <= goal_bias:
            if random.random() <= goal_bias:
                return self.goal
            else:
                rand_nodes = [
                    Node(
                        (
                            random.uniform(0, self.width),
                            random.uniform(0, self.height),
                        )
                    )
                    for _ in range(10)
                ]
                return min(rand_nodes, key=lambda node: node.distance(self.goal))
        else:
            return Node(
                (
                    random.uniform(0, self.width),
                    random.uniform(0, self.height),
                )
            )

    def is_valid_node(self, node: Node) -> bool:
        return (
            0 <= node.x <= self.width
            and 0 <= node.y <= self.height
            and all(node not in obstacle for obstacle in self.obstacles)
        )

    def prob(self, node: Node, spacing: float, steepness: float = 12) -> float:
        if not self.obstacles:
            return 1
        if not self.is_valid_node(node):
            return 0
        min_r = min(obstacle.distance(node) for obstacle in self.obstacles)
        probability = 1 / (1 + math.exp(-steepness * (min_r - spacing)))
        return probability

    def is_suitable_node(
        self, nearest: Node, new_node: Node, spacing: float, steepness: float = 12
    ) -> float:
        if spacing < 0:
            return 1
        if self.is_valid_line(nearest, new_node):
            return self.prob(new_node, spacing, steepness)
        return 0

    def is_valid_line(self, node1: Node, node2: Node) -> bool:
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        for t in np.linspace(0, 1, 10):
            x = node1.x + t * dx
            y = node1.y + t * dy
            if any(Node((x, y)) in obstacle for obstacle in self.obstacles):
                return False

        return True

    def plot(self, view=True, savefig=False):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        plt.scatter(
            self.start.x,
            self.start.y,
            s=100,
            edgecolors="black",
            facecolors="blue",
        )
        plt.scatter(
            self.goal.x,
            self.goal.y,
            s=100,
            edgecolors="black",
            facecolors="red",
        )

        for obstacle in self.obstacles:
            ax.add_patch(
                patches.Rectangle(
                    (obstacle.corner1.x, obstacle.corner1.y),
                    obstacle.width,
                    obstacle.height,
                    facecolor="black" if obstacle.is_wall else "gray",
                )
            )

        details = (
            f"Environment\n\n"
            f"Start: {self.start}\n"
            f"Goal: {self.goal}\n"
            f"Goal Radius: {self.goal_radius}\n"
        )
        plt.title(details)

        fig.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.1)

        if savefig:
            directory = "data/environments"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"{directory}/{len(self.obstacles)}_obstacles.png")
        if view:
            plt.show()
            plt.close()
        else:
            return fig, ax


class RRT:
    def __init__(
        self,
        environment: Environment,
        step_size: float = 2,
        max_iter: int = 10000,
        goal_bias: float = 0.2,
        spacing: float = 1.0,
        steepness: float = 12,
    ):
        self.environment = copy.deepcopy(environment)
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [environment.start]
        self.target = None
        self.goal_bias = goal_bias
        self.spacing = spacing
        self.steepness = steepness

        self.length = 0
        self.path = []

    def reset(self):
        self.nodes = [self.environment.start]
        self.target = None
        self.length = 0
        self.path = []

    def plan(self):
        for _ in range(self.max_iter):
            new_node = self.new()
            self.nodes.append(new_node)

            if self.goal_reached(new_node):
                self.target = new_node
                break

    def new(self) -> Node:
        target = self.environment.random_node(self.goal_bias)
        nearest = self.nearest(target)
        rel_pos = Node((target.x - nearest.x, target.y - nearest.y))
        rel_pos_len = rel_pos.distance(Node((0, 0)))

        step_size = min(self.step_size, rel_pos_len)

        new_pos = Node(
            (
                nearest.x + step_size * rel_pos.x / rel_pos_len,
                nearest.y + step_size * rel_pos.y / rel_pos_len,
            )
        )

        new_node = Node((new_pos.x, new_pos.y), nearest)

        probability = self.environment.is_suitable_node(
            nearest, new_node, self.spacing, self.steepness
        )

        if random.random() <= probability:
            return new_node
        else:
            return self.new()

    def nearest(self, target: Node) -> Node:
        return min(self.nodes, key=lambda node: node.distance(target))

    def goal_reached(self, node: Node) -> bool:
        return node.distance(self.environment.goal) <= self.environment.goal_radius

    def construct(self):
        ptr = self.target
        while ptr is not None:
            self.path.append(ptr)
            self.length += ptr.distance(ptr.parent)
            ptr = ptr.parent
        self.path.reverse()

    def plot(
        self,
        path: Union[list[Node], None] = None,
        smooth: bool = False,
        length: float = 0,
    ):
        if path is None:
            if smooth:
                path, length = self.smoothen_path()
            else:
                path = self.path

        if self.spacing >= 0:
            fig, ax = self.environment.plot(view=False)
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, self.environment.width)
            ax.set_ylim(0, self.environment.height)

        if not smooth:
            for node in self.nodes:
                if node.parent is not None:
                    plt.plot(
                        [node.x, node.parent.x],
                        [node.y, node.parent.y],
                        color="blue",
                        linestyle="dotted",
                        linewidth=1,
                    )
                    plt.scatter(node.x, node.y, s=1, edgecolors="red")

        if self.target is not None:
            for i in range(1, len(path)):
                plt.plot(
                    [path[i].x, path[i - 1].x],
                    [path[i].y, path[i - 1].y],
                    "g-",
                    linewidth=1.5,
                )
                plt.scatter(path[i].x, path[i].y, s=10, edgecolors="black")

        no_of_nodes = len(self.nodes)
        no_of_nodes_in_path = len(path)
        total_length = length if smooth else self.length

        titles = []
        if smooth:
            titles.append("Smooth")
        if self.spacing > 0:
            titles.append("Spaced")
        if self.goal_bias > 0:
            if self.spacing > 0:
                titles.remove("Spaced")
                titles.append("Speedy")
            else:
                titles.append("Greedy")
        if not titles:
            titles = ["Normal"]

        title = " ".join(titles)

        details = (
            rf"$\bf{{{title}}}$"
            f" RRT\n\n"
            f"Step Size: {self.step_size}\n"
            f"Goal Bias: {self.goal_bias}\n"
            f"Spacing: {self.spacing}\n"
            f"Steepness: {self.steepness}\n"
            f"Total Nodes: {no_of_nodes}\n"
            f"Nodes in Path: {no_of_nodes_in_path}\n"
            f"Path Length: {total_length:.2f}"
        )
        plt.title(details)

        fig.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.1)

        plt.show()
        plt.close()

    def smoothen_path(
        self, window_length: int = 7, polyorder: int = 2
    ) -> tuple[list[Node], float]:
        from scipy.signal import savgol_filter

        x = [node.x for node in self.path]
        y = [node.y for node in self.path]

        x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        x_smooth = np.concatenate(([x[0]], x_smooth, [x[-1]]))
        y_smooth = np.concatenate(([y[0]], y_smooth, [y[-1]]))

        smooth_path = [Node((x, y)) for x, y in zip(x_smooth, y_smooth)]
        length = 0
        for i in range(1, len(smooth_path)):
            length += smooth_path[i].distance(smooth_path[i - 1])

        return smooth_path, length

    def plot_prob(self):
        x = np.linspace(0, self.environment.width, 500)
        y = np.linspace(0, self.environment.height, 500)
        x, y = np.meshgrid(x, y)

        z = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                node = Node((x[i, j], y[i, j]))
                z[i, j] = self.environment.prob(node, self.spacing, self.steepness)

        plt.figure(figsize=(6, 5))
        plt.imshow(
            z,
            origin="lower",
            extent=(0, self.environment.width, 0, self.environment.height),
            cmap="Blues_r",
        )
        plt.colorbar(label="Prob")
        plt.title("2D Probability Plot")
        plt.xlabel("x")
        plt.ylabel("y")

        z = 1 - z
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="coolwarm")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("1 - Prob")
        ax.set_zlim(0, 5)
        plt.title("3D Probability Plot")

        plt.show()


def create_env(env_type: int = 2) -> Environment:
    env = Environment(
        (50, 50),
        start=Node((12, 12)),
        goal=Node((36 if env_type == 3 else 38, 38)),
        goal_radius=1,
    )
    env.add_obstacle(Obstacle(Node((0, 0)), Node((10, 50)), is_wall=True))
    env.add_obstacle(Obstacle(Node((40, 0)), Node((50, 50)), is_wall=True))
    env.add_obstacle(Obstacle(Node((10, 0)), Node((40, 10)), is_wall=True))
    env.add_obstacle(Obstacle(Node((10, 40)), Node((40, 50)), is_wall=True))

    if env_type == 2:
        env.add_obstacle(Obstacle(Node((16, 10)), Node((17, 25))))
        env.add_obstacle(Obstacle(Node((25, 16)), Node((26, 34))))
        env.add_obstacle(Obstacle(Node((34, 22)), Node((35, 40))))
    elif env_type == 3:
        env.add_obstacle(Obstacle(Node((15, 15)), Node((16, 35))))
        env.add_obstacle(Obstacle(Node((19, 10)), Node((20, 16))))
        env.add_obstacle(Obstacle(Node((20, 15)), Node((22, 16))))
        env.add_obstacle(Obstacle(Node((16, 21)), Node((20, 22))))
        env.add_obstacle(Obstacle(Node((20, 21)), Node((21, 30))))
        env.add_obstacle(Obstacle(Node((16, 34)), Node((26, 35))))
        env.add_obstacle(Obstacle(Node((25, 15)), Node((26, 35))))
        env.add_obstacle(Obstacle(Node((26, 15)), Node((34, 16))))
        env.add_obstacle(Obstacle(Node((33, 34)), Node((34, 40))))
        env.add_obstacle(Obstacle(Node((34, 34)), Node((37, 35))))
        env.add_obstacle(Obstacle(Node((36, 30)), Node((37, 35))))
        env.add_obstacle(Obstacle(Node((29, 29)), Node((37, 30))))
        env.add_obstacle(Obstacle(Node((29, 21)), Node((30, 30))))
        env.add_obstacle(Obstacle(Node((33, 16)), Node((34, 26))))
        env.add_obstacle(Obstacle(Node((34, 25)), Node((40, 26))))

    return env


def test(rrt_list: list[RRT], env: Environment, no_of_tests: int = 1):
    rrts = []
    if "Normal" in rrt_list:
        normal_rrt = RRT(env, step_size=2, max_iter=10000, goal_bias=0, spacing=0)
        rrts.append(("Normal", normal_rrt))
    if "Spaced" in rrt_list:
        spaced_rrt = RRT(
            env, step_size=2, max_iter=10000, goal_bias=0, spacing=1, steepness=12
        )
        rrts.append(("Spaced", spaced_rrt))
    if "Greedy" in rrt_list:
        greedy_rrt = RRT(
            env, step_size=2, max_iter=10000, goal_bias=0.37, spacing=0, steepness=12
        )
        rrts.append(("Greedy", greedy_rrt))
    if "Speedy" in rrt_list:
        speedy_rrt = RRT(
            env, step_size=2, max_iter=10000, goal_bias=0.37, spacing=1, steepness=12
        )
        rrts.append(("Speedy", speedy_rrt))

    filepath = f"data/{len(env.obstacles)}_obstacles"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(f"{filepath}/rrt_data.csv", "w")
    file.write(
        "RRT,Step Size,Goal Bias,Spacing,Steepness,Path Length,Path Nodes"
        ",Smooth Length,Smooth Nodes,Total Nodes,Time Taken,Obstacles\n"
    )

    no_of_obstacles = len(env.obstacles)

    for _ in range(no_of_tests):
        print(f"\nTest {_ + 1} of {no_of_tests}:")
        for name, rrt in rrts:
            try:
                start = time.time()
                rrt.plan()
                rrt.construct()
                smooth_path, smooth_length = rrt.smoothen_path(
                    window_length=5, polyorder=2
                )
                smooth_nodes = len(smooth_path)
                time_taken = time.time() - start
                print(f"{name} RRT in {time_taken:.2f} seconds.")

                path_length = rrt.length
                path_nodes = len(rrt.path)
                total_nodes = len(rrt.nodes)

                file.write(
                    f"{name},"
                    f"{rrt.step_size},"
                    f"{rrt.goal_bias},"
                    f"{rrt.spacing},"
                    f"{rrt.steepness},"
                    f"{path_length},"
                    f"{path_nodes},"
                    f"{smooth_length},"
                    f"{smooth_nodes},"
                    f"{total_nodes},"
                    f"{time_taken},"
                    f"{no_of_obstacles}\n"
                )
                file.flush()

                rrt.reset()
            except Exception:
                pass

    file.close()


env = create_env(2)
# env.plot(view=True, savefig=False)

rrt = RRT(env, step_size=2, max_iter=10000, goal_bias=0.37, spacing=1, steepness=12)
rrt.plan()
rrt.construct()

rrt.plot()
rrt.plot(smooth=True)
# rrt.plot_prob()

# rrt_list = ("Normal", "Spaced", "Greedy", "Speedy")
# test(rrt_list, env, no_of_tests=1000)
