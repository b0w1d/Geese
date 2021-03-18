from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random

def row_col(position: int, columns: int):
    return position // columns, position % columns

def translate(position: int, direction: Action, columns: int, rows: int):
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column

def adjacent_positions(position: int, columns: int, rows: int):
    return [
        translate(position, action, columns, rows)
        for action in Action
    ]

def min_distance(position: int, food, columns: int):
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )

class GreedyAgent:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation: Observation):
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else random.choice([action for action in Action])
        self.last_action = action

        return action.name

cached_greedy_agents = {}

def greedy_agent(obs, config):
    random.seed(7122)
    index = obs["index"]
    if index not in cached_greedy_agents:
        cached_greedy_agents[index] = GreedyAgent(Configuration(config))
    return cached_greedy_agents[index](Observation(obs))

