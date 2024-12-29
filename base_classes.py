from abc import abstractmethod, ABC
from collections import deque
from torch.nn import Module
from torch.optim import Optimizer


class BaseGame(ABC):
    """
    Base class for the game environment with which the agent interacts.
    """
    def __init__(self):
        self.is_done: bool = False
        self.num_players: int = 2
        self.players = None
        self.current_round: int = 0

    @abstractmethod
    def step(self) -> tuple:
        """
        Step the game given an action towards a new state. Should return new_state, action, reward, and additional
        flags.
        """

    @abstractmethod
    def get_state(self) -> tuple:
        """
        Return the current state of the game. Should contain all the information needed for the agent to learn.
        """
        pass

    def get_actions(self) -> list[int]:
        """
        Return an array of allowed actions.
        """
        pass


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """
    def __init__(self):
        self.batch_size: int = 64
        self.initial_eps: float
        self.final_eps: float
        self.epsilon: float
        self.eps_term: float
        self.gamma: float   # discount factor for Q-learning
        self.tau: float  # factor for soft update of network weights
        self.net_update_freq: int  # soft update frequency
        self.q_local: Module
        self.q_target: Module
        self.optimizer: Optimizer
        self.memory: deque

    @abstractmethod
    def memorize(self, *args) -> None:
        """
        Save a tuple (state, action, reward, next_state, ...) into the agent's memory.
        """
        pass

    @abstractmethod
    def sample(self) -> tuple:
        """
        Sample a batch of saved experiences from replay memory.
        """
        pass

    @abstractmethod
    def update_epsilon(self, current_episode: int) -> None:
        """
        Update the agent's epsilon parameter given the current learning episode.
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load previously saved model weights into the local network.
        """
        pass
