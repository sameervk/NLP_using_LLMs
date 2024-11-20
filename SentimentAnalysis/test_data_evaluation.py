import lightning as L
import torch
import datasets
from lightning.pytorch.loggers import CSVLogger

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoConfig

from local_dataset_utilities import PytorchDatasetDistilBERT
from pytorch_model import LightningModelDistilBERT


def evaluate_test_data(model: AutoModelForSequenceClassification,
                       model_chkpt_path: Path,
                       test_dataloader: torch.utils.data.DataLoader,
                       logger: CSVLogger
                       ):

    # Initialise LightningModel and load weights from checkpoint
    lightning_model = LightningModelDistilBERT.load_from_checkpoint(model_chkpt_path, model=model)

    trainer = L.Trainer(accelerator='cpu',
                        strategy='ddp',
                        devices=6,
                        num_nodes=1,
                        logger=logger
                        )
    # When using more than 1 device, getting the following message
    """
    Using `DistributedSampler` with the dataloaders. During `trainer.test()`, it is recommended to use 
    `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. 
    Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure 
    all devices have same batch size in case of uneven inputs.
    
    However, using 1 device, it takes more time than using 6.
    """

    trainer.test(model=lightning_model, dataloaders=test_dataloader)

    return None


if __name__ == "__main__":

    ##################
    # Import data
    ##################

    current_dir = Path.cwd()

    data_directory = Path.cwd().parent.parent.parent.joinpath('ML datasets/IMDB_SentimentAnalysis_TextData/aclImdb/'
                                                              'processed_data/'
                                                              )
    if not data_directory.exists():
        raise NotADirectoryError("Data Directory does not exist")

    ##############
    # Import tokenized data
    ##############

    # Pretrained ML model name registered on HuggingFace, also required for tokenization
    ml_model = "distilbert-base-uncased"

    # First check the cache folder
    cache_folder = data_directory.joinpath(".cache")
    if not cache_folder.exists():
        Path.mkdir(cache_folder)

    tokenized_data = datasets.load_dataset(path=str(cache_folder),
                                           data_files={"test": "test.arrow"}
                                           )

    # SET FRAMEWORK FOR TRAINING
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # create DataLoader
    # create Dataset first
    test_dataset = PytorchDatasetDistilBERT(huggingface_dataset_dict=tokenized_data, partition="test")

    # DataLoader
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, num_workers=4)

    ##############
    # Initialise Model
    ##############

    model_config = AutoConfig.from_pretrained(ml_model)
    hf_model = AutoModelForSequenceClassification.from_config(config=model_config)

    # Checkpoint folder
    model_chkpt_folder = Path.cwd().joinpath("model_checkpoints/distilBERT_v0")
    model_chkpt_file = "test0-epoch=0-val_acc=0.83.ckpt"

    # Logging folder
    log_folder = Path.cwd().joinpath('logs').joinpath('distilBERT_v0')
    logger = CSVLogger(save_dir=log_folder, name="test0")
    # Initialise Lightning model
    evaluate_test_data(model=hf_model,
                       model_chkpt_path=model_chkpt_folder.joinpath(model_chkpt_file),
                       test_dataloader=test_dataloader,
                       logger=logger
                       )
