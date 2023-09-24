class EnumBase:
    @classmethod
    def get_list(cls):
        return [getattr(cls, attr) for attr in dir(cls) if attr.isupper()]


class Cifar100:
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
