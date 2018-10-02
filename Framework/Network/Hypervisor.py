import tensorflow as tf
import os
import sys
import numpy as np
from typing import List
from .util import Log as log
from .Supervisor import Supervisor
from .NetworkModule import NetworkModule
from ..common import pretty

class Hypervisor(NetworkModule):
    """Wrapper around Supervisor for configuring metaparameters w.r.t training, 
    such as epoch length, how often to save, etc.
    """

    def _setup(self) -> None:
        log.basicConfig(2)
        self.save_dir_abs: str = os.path.join(os.getcwd(),
                self.params['save_dir'])
        if os.path.isdir(self.save_dir_abs):
            log.critical("save_dir:", self.save_dir_abs, "already exists!")
            sys.exit()
        os.makedirs(self.save_dir_abs)
        os.mkdir(os.path.join(self.save_dir_abs, "logs"))
        log_path: str = os.path.join(self.save_dir_abs, "logs", "log")
        log.file_out(log_path)

        run_multiple: bool
        if isinstance(self.params['supervisor_params'], dict):
            # only running one network
            self.params['supervisor_params'] = \
                    [self.params['supervisor_params']]
            log.info("Running single network config")
            run_multiple = False
        elif isinstance(self.params['supervisor_params'], list):
            log.info("Running", len(self.params['supervisor_params']),
                    "network variants")
            run_multiple = True
        else:
            log.critical("supervisor_params must be of type dict or list")
            sys.exit()

        self.run_count: int = 1
        if self.params['run_count'] > 1:
            self.run_count = self.params['run_count']
            log.info("Running", run_count, "trials")

        self.create_supervisors(run_idx = (0 if self.run_count > 1 else None))

        self.pretty = pretty()

        with open(os.path.join(self.save_dir_abs,
            "config.py"), "w+") as f:
            f.write(self.generate_config())

    def create_supervisors(self, run_idx: int = None) -> None:
        """Creates List of supervisors pointing to appropriate output 
        directories.
        """
        if not run_idx is None:
            log.info("Creating supervisors")
        else:
            log.info("Creating supervisors for run", run_idx)

        self.supervisors: List[Supervisor] = []
        super_params = self.params['supervisor_params']
        for i in range(len(super_params)):
            log.info("Creating supervisor", i)
            super_path = os.path.join(self.save_dir_abs, "supervisor" + str(i))
            if not run_idx is None:
                super_path = os.path.join(super_path, str(run_idx))
            super_params[i]['save_dir'] = super_path
            self.supervisors.append(Supervisor(super_params[i],
                parent_params = self.params))

    def run(self) -> None:
        train_batches = \
                int(self.params['train_item_count']/self.params['batchsize'])
        valid_batches = \
                int(self.params['valid_item_count']/self.params['batchsize'])
        test_batches = \
                int(self.params['test_item_count']/self.params['batchsize'])

        epoch_len = self.params['batches_per_epoch']
        step = 0
        for i in range(self.params['epochs']):
            log.info("Epoch", i)
            train_loss = self.supervisor.summarize_epoch_loss(
                    steps = train_batches,
                    global_step = step,
                    mode = Supervisor.TRAIN)
            
            valid_loss = self.supervisor.summarize_epoch_loss(
                    steps = valid_batches,
                    global_step = step,
                    mode = Supervisor.VALID)

            log.info("Training loss:", train_loss)
            log.info("Validation loss:", valid_loss)

            if i % self.params['epochs_to_save'] == 0:
                log.info("Saving at step", step)
                self.supervisor.save(step)

            for j in range(epoch_len):
                step += 1
                self.supervisor.train()
                self.pretty.arrow(j, epoch_len)


        log.info("Running test set")
        test_loss = self.supervisor.summarize_epoch_loss(
                steps = test_batches,
                global_step = step,
                mode = Supervisor.TEST)
        log.info("Test loss:", test_loss)


    def generate_config(self, indent_level: int = 0) -> str:
        conf: str = "from Framework.Network.layers import *\n"
        conf += "from Framework.Network.tf_names import *\n\n"
        conf += "hypervisor_params = {\n"
        idl: int = 1
        sp: int = idl * 4
        conf += self._generate_config(idl, skip = "supervisor_params")

        conf += " "*sp + "'supervisor_params': [\n"
        for supervisor in self.supervisors:
            conf += " "*(sp*2) + "{\n"
            conf += supervisor.generate_config(idl+2)
            conf += " "*(sp*2) + "},\n"
        conf += " "*sp + "]\n"
        conf += "}\n"
        return conf


    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import hypervisor
        return hypervisor
