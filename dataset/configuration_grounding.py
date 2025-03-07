from types import SimpleNamespace
                          
class DatasetConfig(SimpleNamespace):
    def __init__(self,
                 data_path={
                     "/home/zjr2022/datasets/train_vizwiz.json": 1500,
                 },
                 batch_size=2,
                 workers=2,
                 dataset_name=None,
                 ):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            workers=workers,
            dataset_name=dataset_name,
        )

if __name__ == "__main__":
    dataset_config = DatasetConfig()
    print(dataset_config)

