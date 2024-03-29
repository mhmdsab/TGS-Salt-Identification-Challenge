from keras.layers import Activation, Conv2D
from keras.models import Model

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple
from .spatial_squeeze_and_excitation import scSE
from .FPA import Feature_Pyramid_Attention

def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='transpose',
               activation='sigmoid',
               use_batchnorm=True,
               FPA=False,
               SCSE=False):

    input = backbone.input
    x = backbone.output
    
    if FPA:
        x = Feature_Pyramid_Attention(x).FPA()

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < (len(skip_connection_idx)-1):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
        
        if SCSE:
            x = scSE(x,i)._scSE_()
        
        
    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
#    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model
