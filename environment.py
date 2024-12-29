import numpy as np
from configs import BaseGameConfig


class Factory:
    """
    A Factory base class. Represents a pile of 4 mosaic tiles. Only a single tile type can be drawn at a time; the rest
    is discarded onto the central market on the Market.

    Attributes:
        pile (np.array): a numpy array representing 4 mosaic tiles
    """
    def __init__(self):
        self.pile: np.array or None = np.random.randint(1, 6, 4, dtype=np.int8)

    def get_tiles(self, tile_type: int) -> tuple[np.array, np.array]:
        """
        Draws a tile from the pile and discards the remaining ones onto the central market.

        Args:
            tile_type (int): The tile type (color) to be drawn.

        Return:
            draw (np.array): The tiles drawn into the player's hand and discarded tiles put onto the central market
        """

        if self.pile is None:
            raise AttributeError(f"The tile {tile_type} cannot be drawn, as the factory is empty")
        if tile_type not in self.pile:
            raise AttributeError(f"The tile {tile_type} cannot be drawn, as the factory does not contain it")

        mask = self.pile == tile_type
        draw, discard = self.pile[mask], self.pile[~mask]
        self.pile = None

        return draw, discard

    def reset(self):
        self.pile = np.random.randint(1, 6, 4, dtype=np.int8)


class Market(BaseGameConfig):
    """
    Market base class with which a Player can interact to draw tiles from. The market has to be reset before the first
    round and after each successive one.

    Attributes:
        num_factories (int): The number of factories to available. Adjusts automatically to the number of players
            playing the game
        factories (dict): A dictionary encoding factories available on the market.
        central_market (np.array): An array representing the central market. Consists of ints representing available
            tiles types to draw from. Tiles added here are the discarded ones after a Player draws tiles from a factory.
        apply_penalty (bool): Flag, whether a draw from the central market should be punished or not. If True, the
            player drawing tile(s) from the central market receives a penalty = -1.
    """
    def __init__(self, num_players: int):
        super().__init__()
        self.num_players: int = getattr(self, "num_players", num_players)
        assert 4 >= self.num_players >= 2, (f"Number of players must be greater than or equal to 4, "
                                            f"but is {self.num_players}")
        self.num_factories = self.num_players*2 + 1
        self.factories = {i: Factory() for i in range(self.num_factories)}
        self.central_market: np.array = None
        self.apply_penalty: bool = True

    def get_from_factory(self, tile: int, factory_idx: int) -> np.array:
        """
        Draw tiles of one type from a specified factory and discard the remaining ones onto the central market.

        Args:
            tile (int): Tile type to be drawn from a Factory
            factory_idx (int): Index of the Factory to be drawn from

        Return:
            draw (np.array): tiles chosen from a single factory
        """
        factory = self.factories[factory_idx]
        tiles, discard = factory.get_tiles(tile)

        if self.central_market is not None:
            self.central_market = np.append(self.central_market, discard)
        else:
            self.central_market = discard

        return tiles

    def get_from_central_market(self, tile: int) -> tuple[np.array, int]:
        """
        Draw a specified tile from the central market. If it is a first draw from the market, apply a penalty to the
        player drawing. All tiles of the chosen type are drawn onto the player's hand.

        Args:
            tile (int): Tile type to be drawn from the central market
        """
        apply_penalty: int = 0
        assert tile in self.central_market, (
            ValueError(f"The tile {tile} cannot be drawn, as the market does not contain it.")
        )

        if self.apply_penalty:
            apply_penalty = True
            self.apply_penalty = False

        selected_tiles = np.where(self.central_market == tile)
        tiles = self.central_market[selected_tiles]
        self.central_market = np.delete(self.central_market, selected_tiles)

        return tiles, apply_penalty

    def reset(self):
        """
        Refills the factories and resets the central market with its penalty flag. This method should be called
        before a round starts.
        """
        self.factories = {i: Factory() for i in range(self.num_factories)}
        self.central_market: np.array = None
        self.apply_penalty: bool = True


class Player:
    """
    Class representing the player interacting with a Market. The players behaviour has to be implemented step by step
    as follows:
        1. Draw:
            Drawing tiles from the factory or the central market into the player's hand.
        2. Tile placement:
            Placing tiles on the mosaic stack. If the number of tiles in the hand is greater than the number of
            available spaces, the overdrawn tiles add a penalty to the player. Otherwise, the tiles are placed on the
            mosaic stack for decoration later in the game.
        3. Decoration:
            If one horizontal axis of the mosaic stack has been fully filled, the stack is emptied and a single tile
            placed in the mosaic. The player receives points depending on the previous mosaic configuration.

    Note: in the actual game, the penalty and adding of points while filling the mosaic is done at the same time.
    Here both steps are separated; the penalty is applied directly at each filling of the mosaic stack.

    Attributes:
        tiles (np.array): tiles currently held by the player
        total_points(int): the total number of points the player has during the game
        mosaic (np.array): representation of the mosaic to be made. Adding tiles onto it leads to an increase in total
            points of the player
        mosaic_stack (np.array): stack that needs to be filled first in order to place a tile on to the mosaic.
            When an axis of the stack is filled to the full a tile of the same type as in the axis is placed onto the
            mosaic and the axis is reset
        penalties (np.array): array of penalty points added to the player's total score, when the number of tiles
            placed on the mosaic stack is greater than allowable
    """
    def __init__(self):
        tril = np.flip(
            np.tri(5, 5, 0, ), axis=1
        )

        self.is_done: bool = False
        self.mosaic_size: int = 5
        self.total_points: int = 0
        self.tiles: np.array = None
        self.penalties: iter = None
        self.mosaic_stack: np.array = np.where(tril == 0, -1, 0)
        self.mosaic: np.array = self._initialize_mosaic()
        self.available_axes: np.array = np.arange(self.mosaic_size)
        self.unavailable_axes: np.array = np.array([])
        self.reset_penalties()

    def _initialize_mosaic(self, size: int = None) -> np.array:
        """
        Create a square matrix mosaic of dimension size x size. The methods allows for abstraction of the game
        to bigger mosaics.

        Args:
             size (int): the number of rows and columns of the mosaic

        Return:
            mosaic (np.array): array representing the mosaic
        """
        size = self.mosaic_size if size is None else size
        mosaic = np.zeros([size, size], dtype=np.int8)

        for val in range(size):
            zeros = np.zeros([size, size], dtype=np.int8)
            np.fill_diagonal(zeros, val + 1)
            temp = np.roll(zeros, val, 0)
            mosaic += temp

        return mosaic

    def decorate_mosaic(self) -> None:
        """
        Places a tile in the side-stack. If enough tiles are placed along a specific stack, the tile is permanently
        moved to the mosaic stack and gains points based on the current state of the mosaic.
        """
        # 1. check which axes in the stack are already fully filled
        n_stack_axes = np.arange(self.mosaic_size)
        unfilled_stacks = np.argwhere(self.mosaic_stack == 0)
        unfilled_axes = np.unique(unfilled_stacks[:, 0])
        filled_axes_idx = np.setdiff1d(n_stack_axes, unfilled_axes)
        filled_axes = self.mosaic_stack[filled_axes_idx]

        # 2. add a tile to the mosaic if the according stack was fully filled
        for mosaic_col, axis in zip(filled_axes_idx, filled_axes):
            n_tiles_on_axis = len(np.argwhere(axis > 0))
            tile_type = axis[-1].item()
            if n_tiles_on_axis == mosaic_col + 1:
                mosaic_row = np.argwhere(self.mosaic[mosaic_col] == tile_type).item()
                self.mosaic[mosaic_col, mosaic_row] -= tile_type

                # reset the axis of mosaic stack which was filled
                reset_axis = -np.ones(5)
                reset_axis[4-mosaic_col:] = 0
                self.mosaic_stack[mosaic_col] = reset_axis  # reset axis

                # 3. add points
                horizontal, vertical =  self.mosaic[mosaic_col, :], self.mosaic[:, mosaic_row]
                h_count, v_count = 0, 0
                h_idx, v_idx = mosaic_row, mosaic_col.item()
                h_count += self.check_to_right(h_idx, horizontal)
                v_count += self.check_to_right(v_idx, vertical)
                h_count += self.check_to_left(h_idx, horizontal)
                v_count += self.check_to_left(v_idx, vertical)
                points = 1 + h_count + v_count
                self.total_points += points

        # 4. if a row of mosaic is fully filled end the game
        row_is_full = np.all(self.mosaic == 0, axis=1)
        if np.any(row_is_full):
            self.is_done = True

    def place_tiles(self, selected_axis: int):
        """
        Place a specified tiles onto the mosaic stack. This prepares or allows for ``decorate()`` to be called during
        the later stage of the game, after all tiles have been drawn from the Market.

        Args:
            selected_axis (int): axis of the mosaic stack on which the tiles are placed
        """
        # check which rows of the stack are available or already occupied with the same type
        tile_type = np.unique(self.tiles).item()
        self.available_axes = self.get_available_axes(tile_type)
        self.unavailable_axes = self.get_unavailable_axes(tile_type)

        if selected_axis in self.available_axes:
            n_tiles = len(self.tiles)
            current_stack = self.mosaic_stack[selected_axis]
            available_spots = np.argwhere(current_stack == 0)
            overdraw = n_tiles - len(available_spots)
            current_stack[available_spots] = tile_type

            # add penalty if the number of tiles in hand is greater than the available spots in the mosaic stack
            if overdraw > 0:
                try:
                    for penalty, _ in zip(self.penalties, range(overdraw)):
                        self.total_points += penalty
                except StopIteration:
                    # if the penalty stack is empty no further penalties are added
                    self.total_points += 0
            self.tiles = None   # empty the hand for next move

        # if no tiles can be placed on the stack apply penalty for the entire draw
        elif selected_axis in self.unavailable_axes:
            try:
                for penalty, _ in zip(self.penalties, self.tiles):
                    self.total_points += penalty
            except StopIteration:
                self.total_points += 0
            self.tiles = None

        else:
            raise ValueError(f'Unexpected error')

    def get_available_axes(self, tile_type: int) -> np.array:
        """
        Given a tile type check whether it is possible to place the tiles on the mosaic stack. This method takes into
        account, whether the tile type was already placed at a given axis onto the mosaic.

        Args:
             tile_type (int): the tile type to be checked
        """
        unfilled_stacks = np.any(self.mosaic_stack == 0, 1)
        unfilled_mosaic = np.any(self.mosaic == tile_type, 1)

        return np.argwhere(unfilled_stacks & unfilled_mosaic).squeeze()

    def get_unavailable_axes(self, tile_type: int) -> np.array:
        unfilled_mosaic = np.any(self.mosaic == tile_type, 1)
        stack_size = np.arange(self.mosaic_size)

        return np.setdiff1d(stack_size, np.argwhere(unfilled_mosaic).squeeze())

    @staticmethod
    def check_to_right(initial_idx, array: np.array) -> int:
        zeros_count = 0
        while initial_idx < len(array) - 1:
            initial_idx += 1
            if array[initial_idx] == 0:
                zeros_count += 1
            else:
                break

        return zeros_count

    @staticmethod
    def check_to_left(initial_idx, array: np.array) -> int:
        zeros_count = 0
        while initial_idx > 0:
            initial_idx -= 1
            if array[initial_idx] == 0:
                zeros_count += 1
            else:
                break

        return zeros_count

    def draw_tiles_from_factory(self, market: Market, tile: int, factory_idx: int):
        self.tiles = market.get_from_factory(tile, factory_idx)

    def draw_tiles_from_market(self, market: Market, tile: int):
        self.tiles, apply_penalty = market.get_from_central_market(tile)
        self.total_points += next(self.penalties) if apply_penalty else 0

    def reset_penalties(self):
        """
        Reset the penalty generator. Has to be called at the end of each round.
        """
        self.penalties = (p for p in [-1, -1, -2, -2, -2, -3, -3])
