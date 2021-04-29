import numpy as np
from tramp.channels import ColorwiseLinearChannel
import torch

class UpsampleChannel(ColorwiseLinearChannel):
    def __init__(self, input_shape, output_shape, name="Ups"):
        '''
        A linear layer representing bilinear upsampling of 2D images with multiple color channels.

        Parameters
        ----------
        W: colorwise linear transormation.
        input_shape: the shape (c, H1, W1) of input data.
        output_shape: the shape (c, H2, W2) of output data.
        name: name of this operator.

        Notes
        -----
        This implementation is consistent with the Pytorch Upsample operation.
        '''
        n_colors = input_shape[0]
        inp_data_shape = input_shape[1:]
        outp_data_shape = output_shape[1:]

        natural_basis = np.eye(np.prod(inp_data_shape)).reshape((1, -1,) + inp_data_shape)
        operator = torch.nn.Upsample(size=outp_data_shape, mode="bilinear", align_corners=False)
        img_of_operator = operator(torch.FloatTensor(natural_basis))
        img_of_operator = img_of_operator[0].detach().numpy().transpose((1, 2, 0))
        op_matrix = img_of_operator.reshape((np.prod(outp_data_shape), np.prod(inp_data_shape)))

        super().__init__(input_shape, output_shape, W=op_matrix, name=name)



