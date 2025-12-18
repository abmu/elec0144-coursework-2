import copy


class Environment:
    ACTIONS = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    MOVE_PENALTY = -1.0

    # Grid definitions
    START = 'S'
    OBSTACLE = '#'
    TERMINAL = 'T'


    def __init__(self, grid: list[list[str]]) -> None:
        self.grid = copy.deepcopy(grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0
        self.start = self._find_start()
        self.pos = self.start


    def _find_start(self) -> tuple[int]:
        """
        Finds the starting position in the grid

        Returns:
            The starting position
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if self._is_start((i, j)):
                    return (i, j)
        return (0, 0)  # default start
    

    def reset(self) -> None:
        """
        Reset the position to the start position
        """
        self.pos = self.start


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
        if self.is_terminal(self.pos):
            return False, 'Cannot move from a terminal position'
        
        dx, dy = self.ACTIONS[action]
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy

        # Check if within bounds
        if not self._in_bounds((new_x, new_y)):
            return False, 'Move out of bounds'
        
        # Check if obstacle
        if self._is_obstacle((new_x, new_y)):
            return False, 'Cannot move into obstacle'
        
        # Update pos
        self.pos = (new_x, new_y)
        return True, None
    

    def get_pos(self) -> tuple[int, int]:
        """
        Get current position

        Returns:
            A tuple which is the current position
        """
        return self.pos
    

    def _in_bounds(self, pos) -> bool:
        """
        Check if the position is in bounds

        Args:
            pos: Position in the grid

        Returns:
            A boolean
        """
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols
    

    def _is_start(self, pos) -> bool:
        """
        Checks if the position is the start cell

        Args:
            pos: Position in the grid

        Returns:
            A boolean value determining if the position is the start
        """
        cell = self.grid[pos[0]][pos[1]]
        return cell[0] == self.START


    def _is_obstacle(self, pos) -> bool:
        """
        Checks if the position is an obstacle cell

        Args:
            pos: Position in the grid

        Returns:
            A boolean value determining if the position is an obstacle
        """
        cell = self.grid[pos[0]][pos[1]]
        return cell[0] == self.OBSTACLE


    def is_terminal(self, pos) -> bool:
        """
        Checks if the position is a terminal cell

        Args:
            pos: Position in the grid

        Returns:
            A boolean value determining if the position is terminal
        """
        cell = self.grid[pos[0]][pos[1]]
        return cell[0] == self.TERMINAL
    

    def cell_value(self, pos) -> float:
        """
        Get the value of the cell at the input position

        Args:
            pos: Position in the grid

        Returns:
            A float representing the cell value
        """
        if not self._in_bounds(pos) or self._is_obstacle(pos):
            return float('nan')

        cell = self.grid[pos[0]][pos[1]]
        if self._is_start(pos) or self.is_terminal(pos):
            cell = cell[1:]  # remove start/terminal marker character at the beginning
        return float(cell)
    