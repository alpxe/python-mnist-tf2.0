from com.manager.Dataset import Dataset
from com.manager.Download import Download


class Main:
    def __init__(self):
        Download().load()
        dataset = Dataset().read()

        for step, item in enumerate(dataset):
            print(item["label"])
            if step > 10:
                return


if __name__ == "__main__":
    Main()
