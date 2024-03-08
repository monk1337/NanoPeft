import json
import os
import uuid
import torch.nn as nn


class ModelConfigGenerator:
    """
    A class for generating and saving LoRA (Low rank adaptation https://arxiv.org/abs/2106.09685) configuration templates
    for Transformer models. It identifies linear layers in the model and organizes them by their roles
    (e.g., 'query', 'value') for potential LoRA modification.

    Attributes:
        model (torch.nn.Module): The Transformer model to generate a LoRA configuration for.
        config_template (dict): A dictionary mapping layer roles to a boolean indicating whether
                                LoRA should be applied.
        layer_role_mapping (dict): A dictionary mapping layer roles to specific layer paths within the model.
    """

    def __init__(self, model):
        """
        Initializes the LoRAConfigGenerator with a specified model.

        Parameters:
            model (torch.nn.Module): The Transformer model to generate a LoRA configuration for.
        """
        self.model = model
        self.config_template = None
        self.layer_role_mapping = None
        self.generate_simplified_config_with_mapping()
        self.save_config_to_json()


    def generate_simplified_config_with_mapping(self):
        """
        Generates a simplified configuration template and a layer role mapping for the model. The template
        maps layer roles (e.g., 'query', 'value') to booleans for potential LoRA modification, while the mapping
        associates roles with specific layer paths in the model.
        """
        layer_role_mapping = {}

        def traverse_modules(module, path=""):
            """Recursively traverses the model to identify linear layers and their roles."""
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                if isinstance(child, nn.Linear):
                    role = child_path.split(".")[-1]  # Derive role from the layer name
                    if role not in layer_role_mapping:
                        layer_role_mapping[role] = []
                    layer_role_mapping[role].append(child_path)
                else:
                    traverse_modules(child, child_path)

        traverse_modules(self.model)
        self.config_template = {
            role: False for role in layer_role_mapping
        }  # Initialize all roles to False
        self.layer_role_mapping = layer_role_mapping

    def save_config_to_json(
        self, folder_path="nanopeft_configs", default_true_roles=None
    ):

        if default_true_roles is None:
            default_true_roles = ["query", "value"]

        if not hasattr(self, "config_template") or not hasattr(
            self, "layer_role_mapping"
        ):
            self.generate_simplified_config_with_mapping()

        adjusted_config = {
            role: role in default_true_roles for role in self.config_template
        }
        config_with_mapping = {
            "config_template": adjusted_config,
            "layer_role_mapping": self.layer_role_mapping,
        }

        # Check if folder exists, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + "_lora.json"
        full_path = os.path.join(folder_path, unique_filename)

        with open(full_path, "w") as json_file:
            json.dump(config_with_mapping, json_file, indent=4)

        print(f"Configuration saved to {full_path}")
