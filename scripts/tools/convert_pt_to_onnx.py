import torch
import torch.nn as nn
import onnx
import os

# 1. Define the model architecture for both actor and critic
class ActorCritic(nn.Module):
    def __init__(self, observation_space_dim, action_space_dim):
        """
        Initializes the Actor-Critic model with separate networks for the actor (policy)
        and the critic (value function).
        Arguments:
            observation_space_dim (int): Dimension of the observation space, i.e., 124
            action_space_dim (int): Dimension of the action space, i.e., 37
        """
        super(ActorCritic, self).__init__()

        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(observation_space_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space_dim)
        )

        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(observation_space_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        # The forward pass is typically used for training
        # and returns both action distribution parameters and the value estimate.
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value

# --- PyTorch file and input dimensions ---
policy_file_path = "logs/rsl_rl/g1_flat_navigation/2025-09-15_02-57-40/model_11999.pt"
observation_space_dim = 124
action_space_dim = 37

# 2. Instantiate the full ActorCritic model
actor_critic_model = ActorCritic(observation_space_dim, action_space_dim)

# 3. Load the state dictionary from the .pt file
# This loads the trained weights for both actor and critic networks.
try:
    print(f"Loading policy from {policy_file_path}...")
    torch_load = torch.load(policy_file_path)
    torch_load['model_state_dict'].pop('std')
    actor_critic_model.load_state_dict(torch_load['model_state_dict'])
    actor_critic_model.eval()
    print("Policy loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{policy_file_path}' was not found. Please check the path.")
    exit()

# 4. Create a dummy input for tracing
dummy_input = torch.randn(1, observation_space_dim)

# 5. Define the output ONNX file name
output_onnx_path = policy_file_path.replace("pt", "onnx")

# 6. Export only the actor component of the loaded model
print(f"Exporting ONLY the actor (policy) to {output_onnx_path}...")
torch.onnx.export(
    model=actor_critic_model.actor,  # Export only the actor part of the model
    args=dummy_input,
    f=output_onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Export to ONNX complete!")

# 7. Verify the ONNX model file
if os.path.exists(output_onnx_path):
    print(f"\nVerifying the ONNX model at {output_onnx_path}...")
    try:
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except onnx.checker.ValidationError as e:
        print(f"ONNX model is not valid: {e}")