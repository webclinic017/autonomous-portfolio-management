from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset
from azureml.core.compute import ComputeTarget
from azureml.core.datastore import Datastore

# Load the workspace
workspace = Workspace.from_config()

# Get the compute target
compute_target = ComputeTarget(workspace=workspace, name='jdkincan1')

# # # Get the dataset from the datastore
# datastore_path = ('', 'main_data_store_JDKv1.h5')
# dataset = Dataset.File.from_files(path=datastore_path)

# Get the registered dataset from the workspace
dataset = Dataset.get_by_name(workspace, 'main_data_store_JDKv1')


# Define the training environment
env = Environment('lstm-env')
env.python.conda_dependencies.add_pip_package('tensorflow')
env.python.conda_dependencies.add_pip_package('pandas')
env.python.conda_dependencies.add_pip_package('numpy')
env.python.conda_dependencies.add_pip_package('h5py')
env.python.conda_dependencies.add_pip_package('tables')
env.python.conda_dependencies.add_pip_package('requests')
env.python.conda_dependencies.add_pip_package('pathlib')
env.python.conda_dependencies.add_pip_package('matplotlib')
env.python.conda_dependencies.add_pip_package('seaborn')
env.python.conda_dependencies.add_pip_package('scipy')

# Define the script run configuration
src = ScriptRunConfig(source_directory='/home/groovyjac/projects/autonomous-portfolio-management/app/models/LSTM',
                      script='train.py',
                      compute_target=compute_target,
                      environment=env,
                      arguments=['--data-path', dataset.as_named_input('input').as_mount()])

# Submit the experiment
experiment = Experiment(workspace=workspace, name='lstm-training')
run = experiment.submit(src)
run.wait_for_completion(show_output=True)