from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from sagemaker.huggingface import HuggingFace
import boto3
import time
import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
# role = sagemaker.get_execution_role()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

ROLE_NAME = 'AmazonSageMaker-ExecutionRole-20201015T174616'
iam = boto3.client('iam')
role = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']

model_list = ['distilbert-base-uncased', 'distilroberta-base']
s3_prefix_orig = 'samples/datasets/imdb/'

sm = boto3.client('sagemaker')

metric_definitions = [
    {"Name": "test:loss", "Regex": "\'eval_loss\': (.*?),"},
    {"Name": "test:accuracy", "Regex": "\'eval_accuracy\': (.*?),"},
    {"Name": "test:f1", "Regex": "\'eval_f1\': (.*?),"},
    {"Name": "test:precision", "Regex": "\'eval_precision\': (.*?),"},
    {"Name": "test:recall", "Regex": "\'eval_recall\': (.*?),"},
]    


# create SM Experiment
nlp_experiment = Experiment.create(
    experiment_name=f"nlp-classification-{int(time.time())}",
    description="NLP Classification",
    sagemaker_boto_client=sm)

# loop over models
for model_name in model_list:
    trial_name = f"nlp-trial-{model_name}-{int(time.time())}"
    
    # create a trial that will be attached to the experiment
    nlp_trial = Trial.create(
        trial_name=trial_name,
        experiment_name=nlp_experiment.experiment_name,
        sagemaker_boto_client=sm,
    )

    hyperparameters = {'epochs': 2,
                       'train_batch_size': 32,
                       'model_name': model_name
                       }

    huggingface_estimator = HuggingFace(entry_point='train.py',
                                        source_dir='./scripts',
                                        instance_type='ml.p3.2xlarge',
                                        instance_count=1,
                                        role=role,
                                        transformers_version='4.6',
                                        pytorch_version='1.7',
                                        py_version='py36',
                                        hyperparameters=hyperparameters,
                                        metric_definitions=metric_definitions,
                                        enable_sagemaker_metrics=True,)
    
    nlp_training_job_name = f"nlp-training-job-{model_name}-{int(time.time())}"
    
    s3_prefix = s3_prefix_orig + model_name
    training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
    test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
    
    huggingface_estimator.fit(
        inputs={'train': training_input_path, 'test': test_input_path},
        job_name=nlp_training_job_name,
        experiment_config={
            "TrialName": nlp_trial.trial_name,
            "TrialComponentDisplayName": "Training",
        },
        wait=False,
    )
