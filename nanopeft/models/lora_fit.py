import json
import torch.nn as nn
from functools import partial

class LoRAConfigApplier:
    """
    Applies LoRA modifications to a PyTorch model based on a configuration file.
    
    Attributes:
        config_path (str): The file path to the LoRA configuration JSON.
        model (torch.nn.Module): The PyTorch model to be modified.
        LinearWithLoRA (class): The class used to create LoRA-modified linear layers.
        lora_r (int): The rank parameter for the LoRA layers.
        lora_alpha (int): The alpha parameter for the LoRA layers.
        config (dict): Loaded configuration from the JSON file.
    """
    
    def __init__(self, config_path, model, LinearWithLoRA, lora_r=8, lora_alpha=16):
        """
        Initializes the LoRAConfigApplier with model and configuration details.
        
        Parameters:
            config_path (str): Path to the LoRA configuration JSON file.
            model (torch.nn.Module): The model to apply LoRA modifications to.
            LinearWithLoRA (class): The LoRA layer class to use for modifications.
            lora_r (int): Rank parameter for LoRA layers.
            lora_alpha (int): Alpha parameter for LoRA layers.
        """
        self.config_path = config_path
        self.model = model
        for param in model.parameters():
            param.requires_grad = False
        
        self.LinearWithLoRA = LinearWithLoRA
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.config = self._load_config()

    def _load_config(self):
        """Loads the LoRA configuration from the specified JSON file."""
        with open(self.config_path, 'r') as json_file:
            return json.load(json_file)

    def apply_modifications(self):
        """
        Applies the LoRA modifications to the model based on the loaded configuration.
        It replaces specified linear layers with LoRA-enhanced layers according to the config.
        """
        # Create a partial function for creating LoRA layers with specified parameters
        assign_lora = partial(self.LinearWithLoRA, rank=self.lora_r, alpha=self.lora_alpha)
        
        # Iterate over roles specified in the config, applying modifications where indicated
        for role, apply in self.config["config_template"].items():
            if apply:  # Check if modification is requested for the role
                # Apply LoRA to all layers under the current role
                for layer_path in self.config["layer_role_mapping"][role]:
                    self._modify_layer(layer_path, assign_lora)

    def _modify_layer(self, layer_path, assign_lora):
        """
        Modifies a specified layer in the model to use a LoRA layer, if it is a Linear layer.
        
        Parameters:
            layer_path (str): The path to the target layer within the model hierarchy.
            assign_lora (callable): Function to create a LoRA layer with appropriate parameters.
        """
        # Split the path and navigate to the target layer's parent module
        parts = layer_path.split('.')
        module = self.model
        for part in parts[:-1]:  # Exclude the last part, the layer's own name
            module = getattr(module, part)
        
        # Get the target layer by name
        layer_name = parts[-1]
        original_layer = getattr(module, layer_name)
        
        # Replace with a LoRA layer only if it's a Linear layer
        if isinstance(original_layer, nn.Linear):
            setattr(module, layer_name, assign_lora(original_layer))

    @staticmethod
    def count_parameters(model):
        """
        Counts the total number of trainable parameters in the model.
        
        Parameters:
            model (torch.nn.Module): The model for which to count parameters.
            
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example of how to use the class
# model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
# config_path = "path/to/your/config.json"
# lora_applier = LoRAConfigApplier(config_path, model, LinearWithLoRA)
# lora_applier.apply_modifications()
# print("Total number of trainable parameters after LoRA modification:", LoRAConfigApplier.count_parameters(model))
