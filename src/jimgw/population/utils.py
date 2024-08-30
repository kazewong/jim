import importlib

def create_model(model_name):
    try:
        module = importlib.import_module('population_model')

        # Check if model_name is a string
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")

        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}': {str(e)}")
    