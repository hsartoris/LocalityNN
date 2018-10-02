import tensorflow as tf
import os
import numpy as np
from .util import Log as log
from .Supervisor import Supervisor
from .NetworkModule import NetworkModule
from ..common import pretty

class Hypervisor(NetworkModule):
    """Wrapper around Supervisor for configuring metaparameters w.r.t training, 
    such as epoch length, how often to save, etc.
    """

    def _setup(self) -> None:
        self.supervisor: Supervisor = \
            Supervisor(self.params['supervisor_params'],
                    parent_params = self.params)

        self.pretty = pretty()

        with open(os.path.join(self.supervisor.save_dir_abs,
            "config.py"), "w+") as f:
            f.write(self.generate_config())

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

        conf += " "*sp + "'supervisor_params': {\n"
        conf += self.supervisor.generate_config(idl+1)
        conf += " "*sp + "}\n"
        conf += "}\n"
        return conf


    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import hypervisor
        return hypervisor
