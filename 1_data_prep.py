from datasets import load_dataset
from transformers import AutoTokenizer
from datasets.filesystems import S3FileSystem
import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

train_dataset_orig, test_dataset_orig = load_dataset('imdb', split=['train', 'test'])
test_dataset_orig = test_dataset_orig.shuffle().select(range(10000))

model_list = ['distilbert-base-uncased', 'distilroberta-base']
s3_prefix_orig = 'samples/datasets/imdb/'
s3 = S3FileSystem() 

for model_name in model_list:
    print(model_name)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    # tokenize datasets
    train_dataset = train_dataset_orig.map(tokenize, batched=True)
    test_dataset = test_dataset_orig.map(tokenize, batched=True)
    
    # put datasets into torch format
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    s3_prefix = s3_prefix_orig + model_name
    
    # save train_dataset to s3
    training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
    train_dataset.save_to_disk(training_input_path,fs=s3)

    # save test_dataset to s3
    test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
    test_dataset.save_to_disk(test_input_path,fs=s3)
    
print("Data prep done!")
    