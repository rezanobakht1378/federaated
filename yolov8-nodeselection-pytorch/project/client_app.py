"""project: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from project.task import Net, load_data
from project.task import test as test_fn
from project.task import train as train_fn
import random

# Flower ClientApp
app = ClientApp()

# Global dictionaries to track client states across rounds
client_power = {}  # Track battery level for each client (0-100)
client_charging = {}  # Track if client is charging (True/False)
client_computational_power = {}  # Track computational power factor for each client (0-1)
client_id_counter = 0  # Counter to assign unique IDs to clients

# Constants
POWER_DECAY_PER_EPOCH = 5  # درصد کاهش شارژ در هر اپوک
CHARGING_POWER_INCREASE = 10  # درصد افزایش شارژ در حالت شارژ
DELTA_C_DISCHARGING = -0.05  # کاهش قدرت محاسباتی در حالت دشارژ
DELTA_C_CHARGING = +0.1  # افزایش قدرت محاسباتی در حالت شارژ
CHARGING_PROBABILITY = 0.3  # احتمال دلتا سی برای شارژ شدن گوشی

def initialize_client_state(context: Context):
    """Initialize or retrieve client state."""
    global client_id_counter
    
    # Generate unique client ID based on partition and node info
    client_id = f"{context.node_config.get('partition-id', 0)}_{context.node_config.get('node-id', 'unknown')}"
    
    # Initialize client state if not exists
    if client_id not in client_power:
        # Initialize with random power between 1-100
        client_power[client_id] = random.randint(1, 100)
        
        # Initialize computational power (0-1)
        client_computational_power[client_id] = random.uniform(0.5, 1.0)
        
        # Randomly determine if client is charging (30% chance)
        client_charging[client_id] = random.random() < CHARGING_PROBABILITY
        
        client_id_counter += 1
        print(f"Client {client_id} initialized: "
              f"Power={client_power[client_id]:.1f}%, "
              f"Comp Power={client_computational_power[client_id]:.2f}, "
              f"Charging={client_charging[client_id]}")
    
    return client_id

def update_client_power(client_id: str, epochs_used: int):
    """Update client power and computational power after training."""
    if client_id not in client_power:
        return
    
    # Update battery level
    if client_charging[client_id]:
        # Increase power if charging
        power_increase = CHARGING_POWER_INCREASE
        client_power[client_id] = min(100, client_power[client_id] + power_increase)
        
        # Update computational power (increase when charging)
        client_computational_power[client_id] = min(1.0, 
            client_computational_power[client_id] + DELTA_C_CHARGING)
    else:
        # Decrease power based on epochs used
        power_decrease = POWER_DECAY_PER_EPOCH * epochs_used
        client_power[client_id] = max(0, client_power[client_id] - power_decrease)
        
        # Update computational power (decrease when discharging)
        client_computational_power[client_id] = max(0.1, 
            client_computational_power[client_id] + DELTA_C_DISCHARGING)
    
    # Random chance to change charging state for next round
    if random.random() < CHARGING_PROBABILITY:
        client_charging[client_id] = not client_charging[client_id]
    
    print(f"Client {client_id} updated: "
          f"Power={client_power[client_id]:.1f}%, "
          f"Comp Power={client_computational_power[client_id]:.2f}, "
          f"Charging={client_charging[client_id]}")

def adjust_epochs_based_on_power(client_id: str, requested_epochs: int) -> int:
    """Adjust number of epochs based on client's battery level."""
    if client_id not in client_power:
        return requested_epochs
    
    power_percentage = client_power[client_id] / 100.0
    
    # If power is very low (less than 20%), reduce epochs significantly
    if power_percentage < 0.2:
        adjusted_epochs = max(1, int(requested_epochs * 0.3))
    # If power is low (20-50%), reduce epochs moderately
    elif power_percentage < 0.5:
        adjusted_epochs = max(1, int(requested_epochs * 0.6))
    # If power is medium (50-80%), reduce slightly
    elif power_percentage < 0.8:
        adjusted_epochs = max(1, int(requested_epochs * 0.8))
    # If power is high (80-100%), use all requested epochs
    else:
        adjusted_epochs = requested_epochs
    
    # Further adjust based on computational power
    comp_factor = client_computational_power[client_id]
    adjusted_epochs = int(adjusted_epochs * comp_factor)
    
    # Ensure at least 1 epoch
    adjusted_epochs = max(1, adjusted_epochs)
    
    print(f"Client {client_id}: Requested {requested_epochs} epochs, "
          f"adjusted to {adjusted_epochs} (Power: {client_power[client_id]:.1f}%, "
          f"Comp Power: {client_computational_power[client_id]:.2f})")
    
    return adjusted_epochs

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Initialize or retrieve client state
    client_id = initialize_client_state(context)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Get requested epochs and adjust based on power
    requested_epochs = context.run_config.get("local-epochs", 1)
    actual_epochs = adjust_epochs_based_on_power(client_id, requested_epochs)
    
    # Adjust batch size based on computational power if needed
    comp_factor = client_computational_power[client_id]
    adjusted_batch_size = int(batch_size * comp_factor)
    adjusted_batch_size = max(16, adjusted_batch_size)  # Minimum batch size
    
    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        actual_epochs,
        msg.content["config"]["lr"],
        device,
    )

    # Update client power after training
    update_client_power(client_id, actual_epochs)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "client_power": client_power[client_id],
        "client_comp_power": client_computational_power[client_id],
        "client_charging": int(client_charging[client_id]),
        "actual_epochs": actual_epochs,
        "requested_epochs": requested_epochs,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Initialize or retrieve client state
    client_id = initialize_client_state(context)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    
    # Adjust batch size for evaluation based on computational power
    comp_factor = client_computational_power[client_id]
    adjusted_batch_size = int(batch_size * comp_factor)
    adjusted_batch_size = max(16, adjusted_batch_size)
    
    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Update client power after evaluation (smaller impact than training)
    update_client_power(client_id, 1)  # Evaluation counts as 1 "epoch"
    
    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "client_power": client_power[client_id],
        "client_comp_power": client_computational_power[client_id],
        "client_charging": int(client_charging[client_id]),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
