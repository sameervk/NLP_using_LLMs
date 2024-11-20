import torch
import lightning as L
from datasets import load_dataset, Features, Value, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pathlib import Path

from local_dataset_utilities import PytorchDatasetDistilBERT
from pytorch_model import LightningModelDistilBERT, MyProgressBar


def select_input_output_features(input_feature: str, output_feature: str, num_classes: int) -> Features:
    features = Features(dict({input_feature: Value(dtype='string', id=None),
                              output_feature: ClassLabel(num_classes=num_classes)
                              }
                             )
                        )

    return features


def ml_dataset(path: str, data_files: dict, features_to_select: Features) -> DatasetDict:
    return load_dataset(path=path, data_files=data_files, features=features_to_select)


def tokenise_data(ml_model: str, dataset_dict: DatasetDict, cache_folder: Path) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=ml_model)

    cache_files = {"train": "train.arrow",
                   "val": "val.arrow",
                   "test": "test.arrow"
                   }
    check_all_files = [cache_folder.joinpath(cache_files[key]).exists() for key, val in cache_files.items()]

    if all(check_all_files):

        tokenized_data = load_dataset(path=str(cache_folder),
                                      data_files=cache_files
                                      )
        return tokenized_data

    else:

        tokenized_data = dataset_dict.map(lambda x: tokenizer(x["text"], truncation=True, padding=True),
                                          batched=True,
                                          batch_size=None,
                                          cache_file_names={k: str(cache_folder.joinpath(v)) for k, v in
                                                            cache_files.items()}
                                          )

        return tokenized_data


if __name__ == "__main__":

    # import data
    current_dir = Path.cwd()

    data_directory = Path.cwd().parent.parent.parent.joinpath('ML datasets/IMDB_SentimentAnalysis_TextData/aclImdb/'
                                                              'processed_data/'
                                                              )
    if not data_directory.exists():
        raise NotADirectoryError("Data Directory does not exist")

    data_files = {"train": "train.csv",
                  "val": "val.csv",
                  "test": "test.csv"
                  }
    selected_features = select_input_output_features(input_feature="text",
                                                     output_feature="label",
                                                     num_classes=2
                                                     )
    loaded_dataset = ml_dataset(path=str(data_directory),
                                data_files=data_files,
                                features_to_select=selected_features
                                )

    # Pretrained ML model name registered on HuggingFace
    ml_model = "distilbert-base-uncased"

    # ---
    # Tokenize the data

    # First set up cache folder
    cache_folder = data_directory.joinpath(".cache")
    if not cache_folder.exists():
        Path.mkdir(cache_folder)

    tokenized_data = tokenise_data(ml_model=ml_model, dataset_dict=loaded_dataset, cache_folder=cache_folder)

    # SET FRAMEWORK FOR TRAINING
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ---
    # Prepare dataloaders

    # First create pytorch Datasets
    training_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "train")
    val_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "val")
    test_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "test")

    # Now create pytorch DataLoaders
    # SET BATCH SIZE
    batch_size = 32
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset_pytorch,
                                                      batch_size=batch_size,
                                                      num_workers=6,
                                                      shuffle=True
                                                      )
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset_pytorch,
                                                 batch_size=batch_size,
                                                 num_workers=6,
                                                 shuffle=False
                                                 )
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset_pytorch,
                                                  batch_size=batch_size,
                                                  num_workers=6,
                                                  shuffle=False
                                                  )

    # ---
    # Initialise DistilBERT model

    # First create directory to store the model weights
    model_directory = data_directory.parent.parent.parent.parent.joinpath("LLM_models")
    if not model_directory.exists():
        Path.mkdir(model_directory)

    model = AutoModelForSequenceClassification.from_pretrained(ml_model, cache_dir=model_directory, num_labels=2)

    # model appears to be in evaluation mode
    # have to set it to training mode
    model.train()

    # Fine-tuning
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze last two layers
    for param in model.pre_classifier.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Build a Lightning model
    lightning_model = LightningModelDistilBERT(model=model,
                                               learning_rate=5e-5
                                               )

    # ---
    # Add checkpoints

    # Model checkpoint
    modelcheckpoints_dir = Path.cwd().joinpath("model_checkpoints")
    if not modelcheckpoints_dir.exists():
        modelcheckpoints_dir.mkdir()

    folder_name = Path(__file__).name[:-3]
    modelcheckpoints_folder = modelcheckpoints_dir.joinpath(folder_name)
    if not modelcheckpoints_folder.exists():
        modelcheckpoints_folder.mkdir()

    model_checkpoint = ModelCheckpoint(dirpath=modelcheckpoints_folder,
                                       filename="test0-{epoch}-{val_acc:0.2f}",
                                       monitor="val_acc",
                                       mode="max",
                                       save_last=True,
                                       save_top_k=1
                                       )
    # create callbacks list
    callbacks = [model_checkpoint]

    # Progress bar
    progress_bar = MyProgressBar()
    callbacks.append(progress_bar)
    # ---
    # Logger
    log_dir = Path.cwd().joinpath('logs')
    if not log_dir.exists():
        Path.mkdir(log_dir)

    log_folder = log_dir.joinpath(folder_name)
    if not log_folder.exists():
        Path.mkdir(log_folder)

    logger = CSVLogger(save_dir=log_folder, name="test0")

    # ---
    # Lightning Trainer

    try:
        # quick run for testing for bugs
        test_trainer = L.Trainer(accelerator='cpu',
                                 strategy='ddp',
                                 devices=6,
                                 precision="16",
                                 logger=logger,
                                 callbacks=callbacks,
                                 fast_dev_run=True,
                                 )
        test_trainer.fit(lightning_model, train_dataloaders=training_dataloader, val_dataloaders=val_dataloader)

    except Exception as e:
        raise e

    else:
        max_epochs = 1
        trainer = L.Trainer(accelerator='cpu',
                            strategy='ddp',
                            devices=6,
                            precision="16",
                            logger=logger,
                            callbacks=callbacks,
                            max_epochs=max_epochs,
                            )

        trainer.fit(lightning_model, train_dataloaders=training_dataloader, val_dataloaders=val_dataloader)
