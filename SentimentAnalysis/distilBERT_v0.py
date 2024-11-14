import torch
import lightning as L
from datasets import load_dataset, Features, Value, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

from .local_dataset_utilities import PytorchDatasetDistilBERT


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
    cache_folder = data_directory.joinpath("/.cache/")
    if not cache_folder.exists():
        Path.mkdir(cache_folder)

    tokenized_data = tokenise_data(ml_model=ml_model, dataset_dict=loaded_dataset, cache_folder=cache_folder)

    # set framework for training
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ---
    # Prepare dataloaders

    # First create pytorch Datasets
    training_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "train")
    val_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "val")
    test_dataset_pytorch = PytorchDatasetDistilBERT(tokenized_data, "test")

    # Now create pytorch DataLoaders
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset_pytorch, num_workers=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset_pytorch, num_workers=4, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset_pytorch, num_workers=4, shuffle=False)



