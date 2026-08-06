import logging
import sys
import math
import time
from collections import OrderedDict
import tensorboardX


_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # this is the global logger


def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()

    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % asMinutes(s)

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class TensorboardLogger(object):
    def __init__(self):
        self._logger = None
        self._global_step = 0

    def create(self, path):
        self._logger = tensorboardX.SummaryWriter(path)

    def noop(self, *args, **kwargs):
        return

    def step(self):
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    def log_scalar_dict(self, log_dict, prefix=''):
        """ log a dictionary of scalar values"""
        if self._logger is None:
            return
        if prefix:
            prefix = f'{prefix}_'
        for name, value in log_dict.items():
            if isinstance(value, dict):
                self.log_scalar_dict(value, self._global_step,
                                     prefix=f'{prefix}{name}')
            else:
                self._logger.add_scalar(f'{prefix}{name}', value,
                                        self._global_step)

    def __getattr__(self, name):
        if self._logger is None:
            return self.noop
        return self._logger.__getattribute__(name)

TB_LOGGER = TensorboardLogger()


class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """
    def __init__(self, name, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        val = (value if self._val is None
               else value*(1-self._sm) + self._val*self._sm)
        if not math.isnan(val):
            self._val = val

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        if self._val is None:
            return 0
        return self._val

    @property
    def name(self):
        return self._name
