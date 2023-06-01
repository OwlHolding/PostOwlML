import multiprocessing
import os


class RLookPool:

    def __init__(self):
        self.values = {}

    def __getitem__(self, item):
        if not (item in self.values.keys()):
            self.values[item] = multiprocessing.RLock()

        return self.values[item]


class DataBaseException(Exception):
    pass


class DataBase:

    def __init__(self, name: str, main_node: str, child_node: str):
        self.locks = RLookPool()
        self.name = name
        self.main_node = main_node
        self.child_node = child_node

        if not os.path.exists(name):
            os.makedirs(name)

    def set_main_node(self, **kwargs) -> None:
        if not os.path.exists(f'{self.name}/{kwargs[self.main_node]}'):
            os.makedirs(f'{self.name}/{kwargs[self.main_node]}')

    def set_child_node(self, **kwargs) -> None:
        if self.exist_main_node(**kwargs):
            if not os.path.exists(f'{self.name}/{kwargs[self.main_node]}/{kwargs[self.child_node]}'):
                os.makedirs(f'{self.name}/{kwargs[self.main_node]}/{kwargs[self.child_node]}')
        else:
            raise DataBaseException(f"The main_node {kwargs[self.main_node]} is not exist")

    def save_file(self, file_name: str, bfile, **kwargs) -> None:
        with self.locks[kwargs[self.main_node]]:
            with open(f'{self.name}/{kwargs[self.main_node]}/{kwargs[self.child_node]}/{file_name}', 'wb') as file:
                file.write(bfile)

    def load_file(self, file_name: str, **kwargs):
        with self.locks[kwargs[self.main_node]]:
            with open(f'{self.name}/{kwargs[self.main_node]}/{kwargs[self.child_node]}/{file_name}', 'rb') as file:
                text = file.read()

        return text

    def exist_main_node(self, **kwargs):
        if os.path.exists(f'{self.name}/{kwargs[self.main_node]}'):
            return True

        return False

    def exist_child_node(self, **kwargs):
        if os.path.exists(f'{self.name}/{kwargs[self.main_node]}/{kwargs[self.child_node]}'):
            return True

        return False
