import opendatasets
def download_dataset(dataset_url: str, destination: str = 'D:\Christ\T4\LLM\Lab 2\data'):
    """
    Downloads a dataset from Open Datasets.

    Parameters:
    - dataset_url (str): The URL of the dataset to download.
    - destination (str): The directory where the dataset will be saved. Default is current directory.

    Returns:
    - None
    """
    opendatasets.download(dataset_url, data_dir=destination)

download_dataset("https://www.kaggle.com/datasets/chayanonc/1000-folk-stories-around-the-world")