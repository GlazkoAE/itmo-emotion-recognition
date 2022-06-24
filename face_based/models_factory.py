from abc import ABC, abstractmethod

from models import *


class ModelAbstractFactory(ABC):
    @abstractmethod
    def createModel(self, cfg) -> Model:
        pass


class MobileRULFactory(ModelAbstractFactory):
    def createModel(self, cfg) -> Model:
        return MobileRUL(cfg)


def create_factory(model_cgf: str) -> ModelAbstractFactory:
    factory_dict = {"MobileRUL": MobileRULFactory}

    return factory_dict[model_cgf]()
