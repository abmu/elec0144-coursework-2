import copy


class Environment:
    ACTIONS = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    START = 'S'
    OBSTACLE = '#'
    TERMINAL = 'T'


    def __init__(self, grid: list[list[str]]) -> None:
        self.grid = copy.deepcopy(grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0
        self.pos = self._find_start()


    def _find_start(self) -> list[int]:
        """
        Finds the starting position in the grid

        Returns:
            The starting position
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == self.START:
                    return [i, j]
        return [0, 0]  # default start


    def move(self, action: str) -> tuple[bool, str]:
        """
        Move to the new position on the grid based on the action

        Args:
            action: Action to perform

        Returns:
            A tuple containing a boolean indicating whether the move was successful, and an error message
        """
        if action not in self.ACTIONS.keys():
            return False, f'Invalid action. "{action}" is not in allowed actions: {self.ACTIONS.keys()}'
        
        # Check if currently in a terminal state
        if self.is_terminal():
            return False, 'Cannot move from a terminal position'
        
        dx, dy = self.ACTIONS[action]
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy

        # Check if within bounds
        if not (0 <= new_x < self.rows and 0 <= new_y < self.cols):
            return False, 'Move out of bounds'
        
        # Check if obstacle
        if self.grid[new_x][new_y] == self.OBSTACLE:
            return False, 'Cannot move into obstacle'
        
        # Update pos
        self.pos = [new_x, new_y]
        return True, None


    def is_terminal(self) -> bool:
        """
        Checks if the current position is a terminal cell

        Returns:
            A boolean value determining if the current position is terminal
        """
        cell = self.grid[self.pos[0]][self.pos[1]]
        return cell[0] == self.TERMINAL
    

    def cell_value(self) -> float:
        """
        Get the current value of the cell occupied

        Returns:
            A float representing the current cell value
        """
        cell = self.grid[self.pos[0]][self.pos[1]]
        if cell[0] == self.START or cell[0] == self.TERMINAL:
            cell = cell[1:]
        return float(cell)