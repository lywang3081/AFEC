import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == "CUB200":
            return data.CUB200()

        elif name == "ImageNet":
            return data.ImageNet()
