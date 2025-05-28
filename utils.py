import pandas as pd


def load_sbic_dataset(
    train_path="datasets/SBIC_train_processed.csv",
    test_path="datasets/SBIC_test_processed.csv",
):
    # Load train data
    train_data = pd.read_csv(train_path)
    train_texts = train_data["post"].tolist()

    # Extract one-hot encoded labels
    label_columns = [
        "targetCategory_body",
        "targetCategory_culture",
        "targetCategory_disabled",
        "targetCategory_gender",
        "targetCategory_race",
        "targetCategory_social",
        "targetCategory_victim",
    ]
    train_labels = train_data[label_columns].values

    # Load test data
    test_data = pd.read_csv(test_path)
    test_texts = test_data["post"].tolist()
    test_labels = test_data[label_columns].values

    return train_texts, train_labels, test_texts, test_labels


def load_validation_dataset(dev_path="datasets/SBIC_dev_processed.csv"):
    # Load dev data for validation
    dev_data = pd.read_csv(dev_path)
    val_texts = dev_data["post"].tolist()

    # Extract one-hot encoded labels
    label_columns = [
        "targetCategory_body",
        "targetCategory_culture",
        "targetCategory_disabled",
        "targetCategory_gender",
        "targetCategory_race",
        "targetCategory_social",
        "targetCategory_victim",
    ]
    val_labels = dev_data[label_columns].values

    return val_texts, val_labels

# def predict_bias(text):
#     inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
#     outputs = model(inputs)
#     return outputs.numpy()[0]
