import time
import numpy as np
from typing import List, Set, Dict, Any, Callable, Tuple, Type, Optional, Union
from mesa import Agent


class Message:
    """Base class for all communication messages between agents"""
    
    def __init__(self, sender_id: int, timestamp: float = None):
        """
        Initialize a new message.
        
        Args:
            sender_id: Unique ID of the sending agent
            timestamp: Time when message was created (defaults to current time)
        """
        self.sender_id = sender_id
        self.timestamp = timestamp or time.time()
        
    def __str__(self) -> str:
        return f"Message from {self.sender_id} at {self.timestamp}"


class CoverageMessage(Message):
    """Message containing coverage map information"""
    
    def __init__(self, sender_id: int, coverage_data: np.ndarray):
        """
        Initialize a coverage update message.
        
        Args:
            sender_id: Unique ID of the sending agent
            coverage_data: NumPy array representing the coverage map
        """
        super().__init__(sender_id)
        self.coverage_data = coverage_data
        
    def __str__(self) -> str:
        coverage_sum = np.sum(self.coverage_data) if self.coverage_data is not None else 0
        return f"Coverage message from {self.sender_id}: {coverage_sum} cells covered"


class CasualtyMessage(Message):
    """Message containing casualty information"""
    
    def __init__(self, sender_id: int, casualty_positions: Set[Tuple[float, float]]):
        """
        Initialize a casualty report message.
        
        Args:
            sender_id: Unique ID of the sending agent
            casualty_positions: Set of (x,y) positions where casualties were found
        """
        super().__init__(sender_id)
        self.casualty_positions = casualty_positions
        
    def __str__(self) -> str:
        return f"Casualty message from {self.sender_id}: {len(self.casualty_positions)} casualties reported"


class RoleProposalMessage(Message):
    """Message proposing role changes to other agents"""
    
    def __init__(self, sender_id: int, proposed_role: str, reason: str = None):
        """
        Initialize a role proposal message.
        
        Args:
            sender_id: Unique ID of the sending agent
            proposed_role: The role being proposed
            reason: Optional explanation for the proposal
        """
        super().__init__(sender_id)
        self.proposed_role = proposed_role
        self.reason = reason
        
    def __str__(self) -> str:
        return f"Role proposal from {self.sender_id}: {self.proposed_role}" + \
               (f" because {self.reason}" if self.reason else "")


class CommunicationManager:
    """Manages message passing between agents in the simulation"""
    
    def __init__(self, model):
        """
        Initialize the communication manager.
        
        Args:
            model: The simulation model
        """
        self.model = model
        self.message_handlers: Dict[Type[Message], Callable] = {}
        
    def register_handler(self, message_type: Type[Message], handler_function: Callable[[Agent, Message], None]):
        """
        Register a function to handle a specific message type.
        
        Args:
            message_type: The class of message to handle
            handler_function: Function that processes messages of this type
        """
        self.message_handlers[message_type] = handler_function
        
    def create_message(self, sender: Agent, message_type: str, **kwargs) -> Message:
        """
        Create a message of the specified type.
        
        Args:
            sender: The agent sending the message
            message_type: Type of message to create
            **kwargs: Data to include in the message
            
        Returns:
            A message object of the appropriate type
            
        Raises:
            ValueError: If message_type is unknown
        """
        if message_type == "coverage":
            return CoverageMessage(sender.unique_id, kwargs.get("coverage_data"))
        elif message_type == "casualty":
            return CasualtyMessage(sender.unique_id, kwargs.get("casualty_positions"))
        elif message_type == "role":
            return RoleProposalMessage(sender.unique_id, kwargs.get("proposed_role"), kwargs.get("reason"))
        else:
            raise ValueError(f"Unknown message type: {message_type}")
            
    def deliver_messages(self, sender: Agent, messages: List[Message], comm_range: float) -> List[Agent]:
        """
        Deliver messages to all agents within communication range.
        
        Args:
            sender: The agent sending the messages
            messages: List of messages to deliver
            comm_range: Maximum communication range
            
        Returns:
            List of agents that received messages
        """
        if not messages:
            return []
            
        # Check if sender has a position
        if not hasattr(sender, 'pos') or sender.pos is None:
            return []
            
        # Filter for only communicable agents (exclude walls and other non-communicable agents)
        from evac_swarm.agents import RobotAgent, DeploymentAgent
        communicable_agents = [
            agent for agent in self.model.agents 
            if isinstance(agent, (RobotAgent, DeploymentAgent))
        ]
            
        # Get agents within range, passing our filtered list
        try:
            nearby_agents, _ = self.model.space.get_agents_in_radius(
                agent=sender,
                radius=comm_range,
                agent_filter=communicable_agents
            )
        except Exception as e:
            # If there's an error getting nearby agents, return empty list
            print(f"Error getting nearby agents: {e}")
            return []
            
        # If no nearby agents, return empty list
        if not nearby_agents:
            return []
            
        # Filter for line of sight
        recipients = []
        for agent in nearby_agents:
            if agent.unique_id == sender.unique_id:
                continue
                
            # Check line of sight using existing model functionality
            try:
                sender_grid = self.model.space.continuous_to_grid(*sender.pos)
                agent_grid = self.model.space.continuous_to_grid(*agent.pos)
                
                if self.model.is_visible_vectorised(
                    sender_grid, 
                    agent_grid,
                    self.model.space.wall_grid
                ):
                    recipients.append(agent)
                    self.deliver_to_agent(agent, messages)
            except Exception as e:
                # If there's an error checking line of sight, skip this agent
                print(f"Error checking line of sight: {e}")
                continue
                
        return recipients
                
    def deliver_to_agent(self, recipient: Agent, messages: List[Message]) -> None:
        """
        Deliver messages to a specific agent.
        
        Args:
            recipient: The agent receiving the messages
            messages: List of messages to deliver
        """
        for message in messages:
            self.process_message(recipient, message)
                
    def process_message(self, recipient: Agent, message: Message) -> bool:
        """
        Process a received message using the appropriate handler.
        
        Args:
            recipient: The agent receiving the message
            message: The message to process
            
        Returns:
            True if message was processed, False otherwise
        """
        # Find the correct handler based on message type
        for message_type, handler in self.message_handlers.items():
            if isinstance(message, message_type):
                handler(recipient, message)
                return True
                
        return False  # No handler found for this message type


# Example handler functions that could be registered with the CommunicationManager

def handle_coverage_message(agent, message: CoverageMessage) -> None:
    """
    Handle a coverage message by updating the agent's coverage map.
    
    Args:
        agent: The receiving agent
        message: The coverage message
    """
    from evac_swarm.agents import RobotAgent, DeploymentAgent
    
    # Handle robot receiving coverage
    if isinstance(agent, RobotAgent) and agent.coverage is not None and message.coverage_data is not None:
        # Combine coverage maps using logical OR
        agent.coverage = np.logical_or(agent.coverage, message.coverage_data)
    
    # Handle deployment agent receiving coverage
    elif isinstance(agent, DeploymentAgent) and agent.global_coverage is not None and message.coverage_data is not None:
        # Update global coverage map
        agent.global_coverage = np.logical_or(agent.global_coverage, message.coverage_data)
        
        # Track which robot reported
        agent.robots_reported.add(message.sender_id)


def handle_casualty_message(agent, message: CasualtyMessage) -> None:
    """
    Handle a casualty message by updating the agent's reported casualties.
    
    Args:
        agent: The receiving agent
        message: The casualty message
    """
    from evac_swarm.agents import RobotAgent, DeploymentAgent
    
    # Both robot and deployment agents handle casualties the same way
    if hasattr(agent, 'reported_casualties'):
        agent.reported_casualties.update(message.casualty_positions)
        
    # If deployment agent, also track which robot reported
    if isinstance(agent, DeploymentAgent):
        agent.robots_reported.add(message.sender_id)


def handle_role_proposal(agent, message: RoleProposalMessage) -> None:
    """
    Handle a role proposal message.
    
    Args:
        agent: The receiving agent
        message: The role proposal message
    """
    from evac_swarm.agents import RobotAgent
    
    # Only robot agents can change roles
    if isinstance(agent, RobotAgent):
        # This would be implemented based on the agent's role-switching logic
        # For now, just print a debug message
        print(f"Robot {agent.unique_id} received role proposal: {message.proposed_role}")
        # Future implementation could set agent.move_behaviour based on the proposal 