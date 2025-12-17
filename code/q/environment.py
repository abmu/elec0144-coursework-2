import copy


class Environment:
    ACTIONS = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }


    def __init__(self, grid: list[list[str]]) -> None:
        self.grid = copy.deepcopy(grid)
        self.pos = (len(grid)-1, 0)


    def move(self, action: str) -> bool:
        """
        Move to the new position on the grid based on the action

        Args:
            action: Action to perform

        Returns:
            A True or False value depending on if the move was successful
        """
        if action not in self.ACTIONS.keys():
            raise ValueError(f'Invalid action. Must be one of {self.ACTIONS.keys()}')
        