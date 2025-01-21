from datasets import load_dataset
from datasets import load_dataset, DatasetDict


def split_tiny_stories(dataset_name="roneneldan/TinyStories", test_size=0.1, seed=42):

    dataset = load_dataset(dataset_name)

    if "test" not in dataset:
        print(f"Splitting the train set into train and test (test size: {test_size*100}%)...")
        train_test_split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        dataset["train"] = train_test_split["train"]
        dataset["test"] = train_test_split["test"]
    
    return DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"],
        "validation": dataset["validation"]
    })


if __name__ == "__main__":
    data_location = "/fsx/ubuntu/ecoc-llm-env/data/"

    dataset_dict = split_tiny_stories()
    dataset_dict.save_to_disk(data_location)

    print("Save the data at the location : ", data_location)