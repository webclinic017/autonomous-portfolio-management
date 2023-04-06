from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException


def azureml_setup():
    # Load the workspace
    ws = Workspace.from_config()

    # Create a new environment with required packages
    env = Environment("env1-trainingDEC")
    env.python.conda_dependencies = CondaDependencies.create(
        conda_packages=["scikit-learn", "pandas", "numpy", "pathlib", "pytables"],
        pip_packages=["azureml-defaults", "tensorflow", "seaborn", "matplotlib", 'requests', 'chardet', 'charset_normalizer'])

    # Choose a name for your compute target
    compute_name = "computetarget1-trainingDEC"

    try:
        # Check if the compute target already exists
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print("Found existing compute target.")
    except ComputeTargetException:
        # If the compute target doesn't exist, create it
        print("Creating new compute target...")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6",  # Change this to the VM size you want to use
            max_nodes=4  # Change this to the maximum number of nodes you want to use
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    return ws, env, compute_name