from azureml_setup import azureml_setup
from data_preprocessing import load_and_preprocess_data
from dec_model import DECModel, dec_loss, target_distribution, train_dec_model
from sklearn.metrics import silhouette_score
from azureml.core import Experiment, ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling
from azureml.train.hyperdrive.parameter_expressions import choice, loguniform
import tensorflow as tf


if __name__ == "__main__":
    # AzureML Setup
    ws, env, compute_name = azureml_setup()

    # Data Loading and Preprocessing
    data = load_and_preprocess_data()

    # Model Initialization and Compilation
    n_clusters = 10
    n_epochs = 100
    early_stopping_patience = 5
    model = DECModel(n_clusters=n_clusters, input_shape=data.shape[1])
    optimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    # Define a learning rate scheduler function
    def learning_rate_scheduler(epoch):
        initial_lr = 0.01
        decay_rate = 0.9
        decay_steps = 10
        new_lr = initial_lr * (decay_rate ** (epoch // decay_steps))
        return new_lr
    
    model.compile(optimizer=optimizer, loss_fn=dec_loss, learning_rate_scheduler=learning_rate_scheduler)

    # Create a TensorFlow Dataset object
    batch_size = 32
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.shuffle(buffer_size=len(data), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch_size)

    # # Cast the input data to the appropriate data type
    # ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))
    # Train the model
    train_dec_model(model, ds, n_epochs, early_stopping_patience)

    # Evaluate the model
    cluster_assignments = model.cluster_layer(data)
    silhouette = silhouette_score(data, cluster_assignments.numpy().argmax(axis=1))
    print("Silhouette Score: {:.3f}".format(silhouette))

    # Create a new run configuration
    script_config = ScriptRunConfig(
        source_directory="./",
        script="main.py",
        arguments=[],
        compute_target=compute_name,
        environment=env)
    
    # Configure hyperparameter tuning
    ps = RandomParameterSampling({
        '--lr': loguniform(-6, -1),
        '--batch_size': choice(16, 32, 64, 128)
    })

    hd_config = HyperDriveConfig(
        run_config=script_config,
        hyperparameter_sampling=ps,
        primary_metric_name='Silhouette',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=4)

    # Create a new experiment
    experiment = Experiment(workspace=ws, name="experiment4-trainingDEC")

    # Submit the run
    run = experiment.submit(hd_config)

    # Save the model
    tf.keras.models.save_model(model, 'dec_model')

    # Load the model
    loaded_model = tf.keras.models.load_model('dec_model', custom_objects={'DECModel': DECModel})