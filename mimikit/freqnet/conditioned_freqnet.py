import torch.nn as nn
import torch
import numpy as np

from .freqnet import FreqNet
from .modules import GatedLinearInput, AbsLinearOutput, mean_L1_prop
from .freq_layer import FreqLayer
from .base import FreqNetModel, FreqOptim, FreqData
from ..data.data_obj import DataObj, SequenceSlicer
from pytorch_lightning import LightningDataModule


def env_from_stft(spec, att=0.75, rel=0.1):
    lev = 0.0
    sig = np.sum(abs(spec), 1)
    outsig = np.zeros_like(sig)
    for i in range(len(sig)):
        x = sig[i]
        if x > lev:
            lev = lev + att * (x - lev)
        else:
            lev = lev + rel * (x - lev)
        outsig[i] = lev
    return outsig


# how should the dataset thing work
# how should the h5 file be linked to the model
class ConditionedFreqData(FreqData):

    def prepare_data(self, *args, **kwargs):
        if not (getattr(self.model, "targets_shifts_and_lengths", False)):
            raise TypeError("Expected `model` to implement `targets_shifts_and_lengths(input_length)`"
                            " in order to compute the right slices for the batches")
        targets_def = self.model.targets_shifts_and_lengths(self.input_seq_length)
        self.ds = DataObj((SequenceSlicer([(0, self.input_seq_length)] + targets_def, self.ds[0]),
                           SequenceSlicer([(1, self.input_seq_length)], self.ds[1])))

    def get_input_dim(self, data_object):
        return data_object[0].shape[-1]


class ConditionedFreqNet(FreqNet):
    LAYER_KWARGS = ["groups", "strict", "accum_outputs", "concat_outputs",
                    "pad_input", "learn_padding", "with_skip_conv", "with_residual_conv"]

    data_class = ConditionedFreqData

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 model_dim=512,
                 groups=1,
                 n_layers=(2,),
                 strict=False,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 consistency_measure=None,
                 consistency_loss=0.01,
                 condition_bias=False,
                 **data_optim_kwargs):
        super(FreqNet, self).__init__(**data_optim_kwargs)
        self._loss_fn = loss_fn
        self.model_dim = model_dim
        self.groups = groups
        self.n_layers = n_layers
        self.strict = strict
        self.condition_bias = condition_bias
        self.accum_outputs = accum_outputs
        self.concat_outputs = concat_outputs
        self.pad_input = pad_input
        self.learn_padding = learn_padding
        self.with_skip_conv = with_skip_conv
        self.with_residual_conv = with_residual_conv
        self.consistency_loss = consistency_loss
        self.consistency_measure = consistency_measure

        # Input Encoder
        self.inpt = GatedLinearInput(self.input_dim, self.model_dim)
        self.env_inpt = nn.Linear(1, self.model_dim, bias=self.condition_bias)

        # Auto-regressive Part
        layer_kwargs = {attr: getattr(self, attr) for attr in self.LAYER_KWARGS}
        # for simplicity we keep all the layers in a flat list
        self.layers = nn.ModuleList([
            FreqLayer(layer_index=i, input_dim=model_dim, layer_dim=model_dim, **layer_kwargs)
            for n_layers in self.n_layers for i in range(n_layers)
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(self.model_dim, self.input_dim)

        self.save_hyperparameters()

    def forward(self, x, env):
        """
        """
        x = self.inpt(x) + self.env_inpt(env).permute(0, 2, 1)
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips)
        x = self.outpt(skips)
        return x

    def training_step(self, batch, batch_idx):
        batch, target, env = batch
        output = self.forward(batch, env)
        recon = self.loss_fn(output, target)
        # regularization by STFT consistency measure
        if self.consistency_measure is not None:
            consistency = torch.mean(self.consistency_measure(output))
            res = {"loss": recon + self.consistency_loss * consistency, "consistency": consistency}
        else:
            res = {"loss": recon}
        return res

    def validation_step(self, batch, batch_idx):
        batch, target, env = batch
        output = self.forward(batch, env)
        recon = self.loss_fn(output, target)
        return {"val_loss": recon}


