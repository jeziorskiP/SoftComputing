import argparse


class ArgumentParser:
    args = None

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='ADZ - Etap 1', formatter_class=argparse.RawTextHelpFormatter,
            description='Politechnika Łódzka\
                         \nAnaliza Danych Złożonych - Etap 1\
                         \n\nAutorzy:\
                         \n  Paweł Jeziorski\t234066')

        self.parser.add_argument('-lr', metavar='N', dest='learning_rate',
                                 type=float, default=0.001,
                                 help='learning_rate')

        self.parser.add_argument('-e', metavar='N', dest='acceptable_error',
                                 type=float, default=0.01,
                                 help='acceptable_error - stop condition')
                                 
        self.parser.add_argument('-m', metavar='N', dest='momentum',
                                 type=float, default=0.0001,
                                 help='momentum')

        self.parser.add_argument('-hs', metavar='N', dest='hidden_size',
                                 type=int, default=2,
                                 help='hidden_size - define hiiden size')

        self.parser.add_argument('-b', metavar='DELTA', dest='bias', type=int,
                                 default=1, help='bias - 1-> yes, 0-> no')

        self.args = self.parser.parse_args()

    def get_acceptable_error(self):
        return self.args.acceptable_error
        
    def get_momentum(self):
        return self.args.momentum
        
    def get_learning_rate(self):
        return self.args.learning_rate
        
    def get_hidden_size(self):
        return self.args.hidden_size

    def get_bias(self):
        return self.args.bias