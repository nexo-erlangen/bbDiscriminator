from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, LocallyConnected1D, LocallyConnected2D
from keras.layers import Flatten, Dropout, BatchNormalization, Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate, Add
from keras import regularizers
from keras import layers
from keras import backend as K
import copy

def create_shared_dcnn_network_2():
    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'kernel_initializer': 'glorot_uniform'}
              # 'kernel_regularizer': regularizers.l2(1.e-2)}

    input = []
    input.append(Input(shape=(350, 38, 1), name='Wire_1'))
    input.append(Input(shape=(350, 38, 1), name='Wire_2'))

    layers = []
    layers.append(Conv_block(16, filter_size=(5, 3), name='ConvBl_1', max_pooling=None, **kwargs))
    layers.append(Conv_block(16, filter_size=(5, 3), name='ConvBl_2', max_pooling=(4, 2), **kwargs))
    layers.append(Conv_block(32, filter_size=(5, 3), name='ConvBl_3', max_pooling=None, **kwargs))
    layers.append(Conv_block(32, filter_size=(5, 3), name='ConvBl_4', max_pooling=(4, 2), **kwargs))
    layers.append(Conv_block(64, filter_size=(3, 3), name='ConvBl_5', max_pooling=None, **kwargs))
    layers.append(Conv_block(64, filter_size=(3, 3), name='ConvBl_6', max_pooling=(2, 2), **kwargs))
    layers.append(Conv_block(128, filter_size=(3, 3), name='ConvBl_7', max_pooling=None, **kwargs))
    layers.append(Conv_block(128, filter_size=(3, 3), name='ConvBl_8', max_pooling=(2, 2), **kwargs))
    layers.append(Conv_block(256, filter_size=(3, 3), name='ConvBl_9', max_pooling=None, **kwargs))
    #TEST
    layers.append(Conv_block(256, filter_size=(3, 3), name='ConvBl_10', max_pooling=None, **kwargs))
    layers.append([GlobalAveragePooling2D()])
    #TEST
    # layers.append(Conv_block(256, k_size=(3, 3), padding=padding, init=init, dropout=drop, max_pooling=(2, 2), activ=act, kernel_reg=regu))
    # layers.append([Flatten()])
    # layers.append([Dense(64, activation='relu', kernel_initializer=kwargs['kernel_initializer'])]) #, kernel_regularizer=regu)])
    # layers.append([Dense(32, activation='relu', kernel_initializer=kwargs['kernel_initializer'])]) #, kernel_regularizer=regu)])

    paths = assemble_network(input, layers)

    merge = Concatenate(name='Flat_1_and_2')(paths)
    # merge = paths[0]

    output = Dense(2, name='Output', activation='softmax', kernel_initializer=kwargs['kernel_initializer'])(merge)
    return Model(inputs=input, outputs=output)

def create_shared_dcnn_network_4():

    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': False,
              'kernel_initializer': 'glorot_uniform',
              'kernel_regularizer': regularizers.l2(1.e-2)}

    inputU = []
    inputU.append(Input(shape=(350, 38, 1), name='U-Wire_1'))
    inputU.append(Input(shape=(350, 38, 1), name='U-Wire_2'))
    inputV = []
    inputV.append(Input(shape=(350, 38, 1), name='V-Wire_1'))
    inputV.append(Input(shape=(350, 38, 1), name='V-Wire_2'))

    layersU = []
    layersV = []
    for layers in [layersU, layersV]: # Use same architecture for U and V wires. Can be changed
        layers.append(Conv_block(16, filter_size=(5, 3), max_pooling=None, **kwargs))
        layers.append(Conv_block(16, filter_size=(5, 3), max_pooling=(4, 2), **kwargs))
        layers.append(Conv_block(32, filter_size=(5, 3), max_pooling=None, **kwargs))
        layers.append(Conv_block(32, filter_size=(5, 3), max_pooling=(4, 2), **kwargs))
        layers.append(Conv_block(64, filter_size=(3, 3), max_pooling=None, **kwargs))
        layers.append(Conv_block(64, filter_size=(3, 3), max_pooling=(2, 2), **kwargs))
        layers.append(Conv_block(128, filter_size=(3, 3), max_pooling=None, **kwargs))
        layers.append(Conv_block(128, filter_size=(3, 3), max_pooling=(2, 2), **kwargs))
        layers.append(Conv_block(256, filter_size=(3, 3), max_pooling=None, **kwargs))
        layers.append(Conv_block(256, filter_size=(3, 3), max_pooling=None, **kwargs))
        layers.append([GlobalAveragePooling2D()])

    pathsU = assemble_network(inputU, layersU)
    pathsV = assemble_network(inputV, layersV)

    inputUV = []
    inputUV.append(Concatenate(name='TPC_1')([pathsU[0], pathsV[0]]))
    inputUV.append(Concatenate(name='TPC_2')([pathsU[1], pathsV[1]]))

    layersUV = []
    layersUV.append([Dense(32, activation='relu', kernel_regularizer=kwargs['kernel_regularizer'])])
    layersUV.append([Dense(16, activation='relu', kernel_regularizer=kwargs['kernel_regularizer'])])

    pathsUV = assemble_network(inputUV, layersUV)

    merge = Concatenate(name='Flat_1_and_2')(pathsUV)
    output = Dense(2, name='Output', activation='softmax', kernel_initializer=kwargs['kernel_initializer'])(merge)

    inputUV = []
    inputUV.append(inputU[0])
    inputUV.append(inputV[0])
    inputUV.append(inputU[1])
    inputUV.append(inputV[1])

    return Model(inputs=inputUV, outputs=output)

def create_shared_inception_network_2_extra_input(kwargs_inc={}):
    kwargs_out = dict(kwargs_inc)
    for key in ['dropout', 'activity_regularizer', 'kernel_regularizer']:
        kwargs_out.pop(key, None)

    # auxiliary_input = Input(shape=(10, 4), name='Aux_Input')
    # x = LocallyConnected1D(10, kernel_size=1, activation='relu', name='21', **kwargs_inc)(auxiliary_input)
    # x = LocallyConnected1D(20, kernel_size=1, activation='relu', name='22', **kwargs_inc)(x)
    # # x = LocallyConnected1D(40, kernel_size=1, activation='relu', name='21')(x)
    # merge_pos = Flatten(name='Flatten_Pos')(x)
    # output_pos = Dense(2, name='Output_Pos', activation='softmax', kernel_initializer="glorot_uniform", **kwargs_out)(merge_pos)
    # # return Model(inputs=[auxiliary_input], outputs=[output_pos])

    # auxiliary_input = Input(shape=(10, 4, 1), name='aux_input')
    # x = Conv2D(10, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_normal")(auxiliary_input)
    # # # x = Conv2D(10, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_uniform")(x)
    # # # x = Conv2D(10, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_uniform")(x)
    # # # x = Conv2D(20, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_uniform")(x)
    # x = Conv2D(20, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_normal")(x)
    # x = Conv2D(40, kernel_size=(1, 4), padding='same', activation='relu', kernel_initializer="glorot_normal")(x)
    # merge_pos = Flatten()(x)
    # # # merge_pos = Dense(56, activation='relu', kernel_initializer="glorot_uniform")(x)
    # output_pos = Dense(2, name='Output_Pos', activation='softmax', kernel_initializer="glorot_normal")(merge_pos)
    # return Model(inputs=[auxiliary_input], outputs=[output_pos])

    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'kernel_initializer': 'glorot_uniform'}
    kwargs = merge_two_dicts(kwargs, kwargs_inc)

    input = []
    input.append(Input(shape=(350, 38, 1), name='Wire_1'))
    input.append(Input(shape=(350, 38, 1), name='Wire_2'))

    layers = []
    layers.append(Conv_block(32, filter_size=(3, 3), name='ConvBl_1', **kwargs))
    layers.append(Conv_block(32, filter_size=(3, 3), name='ConvBl_2', **kwargs))
    layers.append([MaxPooling2D((4, 2), name='maxp_1')])
    layers.append(Conv_block(64, filter_size=(3, 3), name='ConvBl_3', **kwargs))

    num_filters = (64, (96, 128), (16, 32), 32) #TODO Real values from IncV1 Paper?
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_1', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_2', kwargs_inc=kwargs_inc))
    layers.append([MaxPooling2D((2, 2), name='maxp_2')]) # TODO Test Inception module with stride=2 instead of max pooling
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_3', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_4', kwargs_inc=kwargs_inc))
    layers.append([MaxPooling2D((2, 1), name='maxp_3')])
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_5', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_6', kwargs_inc=kwargs_inc))
    layers.append([MaxPooling2D((2, 1), name='maxp_4')])
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_7', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_8', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_9', kwargs_inc=kwargs_inc))
    layers.append(InceptionV1_block(num_filters=num_filters, name='IncBl_10', kwargs_inc=kwargs_inc))
    layers.append([GlobalAveragePooling2D(name='GAverPool_Top')])

    paths = assemble_network(input, layers)
    merge_top = Concatenate(name='Concat_Top_1_and_2')(paths)

    output_top = Dense(2, name='Output_Top', activation='softmax', kernel_initializer="glorot_uniform", **kwargs_out)(merge_top)
    # return Model(inputs=input, outputs=output_top)

    auxiliary_input = Input(shape=(10, 4), name='Aux_Input')
    auxiliary_flat = Flatten(name='Flatten_Aux_Input')(auxiliary_input)

    x = Concatenate(name='Merge_Pos_and_Top')([merge_top, auxiliary_flat])
    # x = Dense(64, activation='relu', kernel_initializer="glorot_uniform", name='31')(x)
    # x = Dense(64, activation='relu', kernel_initializer="glorot_uniform", name='32')(x)
    # x = Dense(64, activation='relu', kernel_initializer="glorot_uniform", name='33')(x)
    x = Dense(256, activation='relu', kernel_initializer="glorot_uniform", name='31')(x)
    x = Dropout(0.3, name='drop_1')(x)
    x = Dense(64, activation='relu', kernel_initializer="glorot_uniform", name='32')(x)
    x = Dropout(0.3, name='drop_2')(x)
    x = Dense(16, activation='relu', kernel_initializer="glorot_uniform", name='33')(x)
    output = Dense(2, name='Output', activation='softmax', kernel_initializer="glorot_uniform")(x)

    return Model(inputs=[input[0], input[1], auxiliary_input], outputs=[output, output_top])

def create_shared_inception_network_2():
    # kwargs = {'padding': 'same',
    #           'dropout': 0.2, #TODO TEST DROPOUT
    #           'BN': True,
    #           'kernel_initializer': 'glorot_uniform'}
    #
    # input = []
    # input.append(Input(shape=(350, 38, 1), name='Wire_1'))
    # input.append(Input(shape=(350, 38, 1), name='Wire_2'))
    #
    # layers = []
    # layers.append(Conv_block(32, filter_size=(3, 3), name='1', **kwargs))
    # layers.append(Conv_block(32, filter_size=(3, 3), name='2', **kwargs))
    # layers.append([MaxPooling2D((4, 2), name='3')])
    # layers.append(Conv_block(64, filter_size=(3, 3), name='4', **kwargs))
    #
    # num_filters = (64, (96, 128), (16, 32), 32) #TODO Real values from IncV1 Paper?
    # layers.append(InceptionV1_block(num_filters=num_filters, name='5'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='6'))
    # layers.append([MaxPooling2D((2, 2), name='7')]) # TODO Test Inception module with stride=2 instead of max pooling
    # layers.append(InceptionV1_block(num_filters=num_filters, name='8'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='9'))
    # layers.append([MaxPooling2D((2, 1), name='10')])
    # layers.append(InceptionV1_block(num_filters=num_filters, name='11'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='12'))
    # layers.append([MaxPooling2D((2, 1), name='13')])
    # layers.append(InceptionV1_block(num_filters=num_filters, name='14'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='15'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='16'))
    # layers.append(InceptionV1_block(num_filters=num_filters, name='17'))
    #
    # layers.append([GlobalAveragePooling2D(name='18')])
    #
    # paths = assemble_network(input, layers)
    #
    # merge = Concatenate(name='Conc_Top_1_and_2')(paths)
    # output = Dense(2, name='Output_Top', activation='softmax', kernel_initializer="glorot_uniform")(merge)
    #
    # return Model(inputs=input, outputs=output)

    # TODO Baseline below

    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'kernel_initializer': 'glorot_uniform'}

    input = []
    input.append(Input(shape=(350, 38, 1), name='Wire_1'))
    input.append(Input(shape=(350, 38, 1), name='Wire_2'))

    layers = []
    layers.append(Conv_block(32, filter_size=(3, 3), name='1', **kwargs))
    layers.append(Conv_block(32, filter_size=(3, 3), name='2', **kwargs))
    layers.append([MaxPooling2D((4, 2), name='3')])
    layers.append(Conv_block(64, filter_size=(3, 3), name='4', **kwargs))

    num_filters = (64, (96, 128), (16, 32), 32) #TODO Real values from IncV1 Paper?
    layers.append(InceptionV1_block(num_filters=num_filters, name='5'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='6'))
    layers.append([MaxPooling2D((2, 2), name='7')]) # TODO Test Inception module with stride=2 instead of max pooling
    layers.append(InceptionV1_block(num_filters=num_filters, name='8'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='9'))
    layers.append([MaxPooling2D((2, 1), name='10')])
    layers.append(InceptionV1_block(num_filters=num_filters, name='11'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='12'))
    layers.append([MaxPooling2D((2, 1), name='13')])
    layers.append(InceptionV1_block(num_filters=num_filters, name='14'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='15'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='16'))
    layers.append(InceptionV1_block(num_filters=num_filters, name='17'))

    layers.append([GlobalAveragePooling2D(name='18')])

    paths = assemble_network(input, layers)

    merge = Concatenate(name='Conc_Top_1_and_2')(paths)
    output = Dense(2, name='Output_Top', activation='softmax', kernel_initializer="glorot_uniform")(merge)

    return Model(inputs=input, outputs=output)

def create_shared_inceptionV4_network_2():
    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'kernel_initializer': 'glorot_uniform'}

    input = []
    input.append(Input(shape=(350, 38, 1), name='Wire_1'))
    input.append(Input(shape=(350, 38, 1), name='Wire_2'))

    layers = []
    layers.append(Conv_block(32, filter_size=(3, 3), name='1', **kwargs))
    layers.append(Conv_block(32, filter_size=(3, 3), name='2', **kwargs))
    layers.append([MaxPooling2D((4, 1), name='3')]) #(4,2)
    layers.append(Conv_block(64, filter_size=(3, 3), name='4', **kwargs))

    num_filters = (96, (64, 96), (64, 96), 96) #TODO Real values from IncV1 Paper?
    layers.append(InceptionV4_block(num_filters=num_filters, name='5'))
    layers.append(InceptionV4_block(num_filters=num_filters, name='6'))
    layers.append([MaxPooling2D((2, 2), name='7')])
    layers.append(InceptionV4_block(num_filters=num_filters, name='8'))
    layers.append(InceptionV4_block(num_filters=num_filters, name='9'))
    layers.append([MaxPooling2D((2, 1), name='10')])
    layers.append(InceptionV4_block(num_filters=num_filters, name='11'))
    layers.append(InceptionV4_block(num_filters=num_filters, name='12'))

    layers.append([GlobalAveragePooling2D(name='18')])

    paths = assemble_network(input, layers)

    merge = Concatenate(name='Conc_Top_1_and_2')(paths)
    output = Dense(2, name='Output_Top', activation='softmax', kernel_initializer="glorot_uniform")(merge)

    return Model(inputs=input, outputs=output)

def create_shared_inception_network_4():
    kwargs = {'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'kernel_initializer': 'glorot_uniform'}

    inputU = []
    inputU.append(Input(shape=(350, 38, 1), name='U-Wire_1'))
    inputU.append(Input(shape=(350, 38, 1), name='U-Wire_2'))
    inputV = []
    inputV.append(Input(shape=(350, 38, 1), name='V-Wire_1'))
    inputV.append(Input(shape=(350, 38, 1), name='V-Wire_2'))

    layersU = []
    layersV = []
    for layers in [layersU, layersV]:  # Use same architecture for U and V wires. Can be changed
        layers.append(Conv_block(32, filter_size=(3, 3), **kwargs))
        layers.append(Conv_block(32, filter_size=(3, 3), **kwargs))
        layers.append([MaxPooling2D((4, 2))])
        layers.append(Conv_block(64, filter_size=(3, 3), **kwargs))

        num_filters = (64, (96, 128), (16, 32), 32) #TODO Real values from IncV1 Paper?
        num_filters_deep = (96, (96, 192), (64, 96), 96)
        layers.append(InceptionV1_block(num_filters=num_filters))
        layers.append(InceptionV1_block(num_filters=num_filters))
        layers.append([MaxPooling2D((2, 2))]) # TODO Test Inception module with stride=2 instead of max pooling
        layers.append(InceptionV1_block(num_filters=num_filters))
        layers.append(InceptionV1_block(num_filters=num_filters))
        layers.append([MaxPooling2D((2, 1))])
        layers.append(InceptionV1_block(num_filters=num_filters))
        layers.append(InceptionV1_block(num_filters=num_filters))
        # layers.append([MaxPooling2D((2, 1))])
        # layers.append(InceptionV1_block(num_filters=num_filters))
        # layers.append(InceptionV1_block(num_filters=num_filters))
        # layers.append(InceptionV1_block(num_filters=num_filters_deep))
        # layers.append(InceptionV1_block(num_filters=num_filters_deep))
        layers.append([GlobalAveragePooling2D()])


    pathsU = assemble_network(inputU, layersU)
    pathsV = assemble_network(inputV, layersV)

    inputUV = []
    inputUV.append(Concatenate(name='TPC_1')([pathsU[0], pathsV[0]]))
    inputUV.append(Concatenate(name='TPC_2')([pathsU[1], pathsV[1]]))

    layersUV = []
    layersUV.append([Dense(64, activation='relu')])
    layersUV.append([Dense(16, activation='relu')])

    pathsUV = assemble_network(inputUV, layersUV)

    merge = Concatenate(name='Flat_1_and_2')(pathsUV)
    output = Dense(2, name='Output', activation='softmax', kernel_initializer="glorot_uniform")(merge)

    inputUV = []
    inputUV.append(inputU[0])
    inputUV.append(inputV[0])
    inputUV.append(inputU[1])
    inputUV.append(inputV[1])

    return Model(inputs=inputUV, outputs=output)

def Conv_block(num_filters, filter_size=(3,3), max_pooling=None, padding='same', dropout=0.0, BN=False, name='', **kwargs):
    """
    2D Convolutional block followed by optional BatchNormalization, Activation (not optional), MaxPooling or Dropout.
    C-(BN)-A-(MP)-(D)
    :param int n_filters: Number of filters used for the convolution.
    :param tuple k_size: Kernel size which is used for all three dimensions.
    :param None/tuple max_pooling: Specifies if a MaxPooling layer should be added. e.g. (1,1,2) for 3D.
    :param string padding: Specifies padding scheme for Conv2D and for MaxPooling. valid/same.
    :param float dropout: Adds a dropout layer if value is greater than 0.
    :param bool BN: Specifies whether to use BatchNormalization or not.
    :param **kwargs: Additional arguments for calling the Conv2D function.
    :return: x: List of resulting output layers.
    """

    x = []
    if BN == False:
        x.append(Conv2D(num_filters, kernel_size=filter_size, padding=padding, activation='relu', name='conv_%s'%(name), **kwargs))
    else:
        channel_axis = -1 if K.image_data_format() == "channels_last" else 1
        x.append(Conv2D(num_filters, kernel_size=filter_size, padding=padding, name='conv_%s'%(name), **kwargs))
        x.append(BatchNormalization(axis=channel_axis, momentum=0.5, scale=False, name='bn_%s'%(name))) #momentum=0.99 #TODO use 0.5 again
        x.append(Activation('relu', name='act_%s'%(name)))

    if max_pooling is not None:
        x.append(MaxPooling2D(strides=max_pooling, padding=padding, name='maxp_%s'%(name)))
    if dropout > 0.0:
        # x.append(Dropout(dropout, name='drop_%s'%(name)))
        x.append(SpatialDropout2D(dropout, name='drop_%s'%(name)))
    return x

def InceptionV1_block(num_filters=(64, (64, 96), (48, 64), 32), name='', kwargs_inc={}):
    """
    2D Inception V1 block. Each Conv2D Element consists of Conv2D, BatchNorm, Activation.
    Inception towers are: 1x1, 1x1 + 3x3, 1x1 + 5x5, AverPool + 1x1
    :param tuple num_filters: Kernel sizes which are used for the tower. Ordering like above.
    :return: x: List of resulting output layers. Concat layer is parsed as last tower.
    """
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1
    kwargs = {'max_pooling': None,
              'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'strides': 1,
              'use_bias': False}
    kwargs = merge_two_dicts(kwargs, kwargs_inc)

    branch1x1 = []
    branch1x1.append(Conv_block(num_filters[0], (1, 1), name='1x1_%s'%(name), **kwargs))
    branch1x1 = sum(branch1x1, [])

    branch3x3 = []
    branch3x3.append(Conv_block(num_filters[1][0], (1, 1), name='3x3-1_%s'%(name), **kwargs))
    branch3x3.append(Conv_block(num_filters[1][1], (3, 3), name='3x3-2_%s'%(name), **kwargs))
    branch3x3 = sum(branch3x3, [])

    branch5x5 = []
    branch5x5.append(Conv_block(num_filters[2][0], (1, 1), name='5x5-1_%s'%(name), **kwargs))
    branch5x5.append(Conv_block(num_filters[2][1], (5, 5), name='5x5-2_%s'%(name), **kwargs))
    branch5x5 = sum(branch5x5, [])

    branch_pool = []
    branch_pool.append([MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='maxp_pool_%s'%(name))])
    branch_pool.append(Conv_block(num_filters[3], (1, 1), name='conv_pool_%s'%(name), **kwargs))
    branch_pool = sum(branch_pool, [])

    concat = Concatenate(axis=channel_axis, name='conc_%s'%(name))

    return [branch1x1, branch3x3, branch5x5, branch_pool, concat]

def InceptionV4_block(num_filters=(96, (64, 96), (64, 96), 96), name='', kwargs_inc={}):
    """
    2D Inception V1 block. Each Conv2D Element consists of Conv2D, BatchNorm, Activation.
    Inception towers are: 1x1, 1x1 + 3x3, 1x1 + 5x5, AverPool + 1x1
    :param tuple num_filters: Kernel sizes which are used for the tower. Ordering like above.
    :return: x: List of resulting output layers. Concat layer is parsed as last tower.
    """
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1
    kwargs = {'max_pooling': None,
              'padding': 'same',
              'dropout': 0.0,
              'BN': True,
              'strides': 1,
              'use_bias': False}
    kwargs = merge_two_dicts(kwargs, kwargs_inc)

    branch1x1 = []
    branch1x1.append(Conv_block(num_filters[0], (1, 1), name='1x1_%s'%(name), **kwargs))
    branch1x1 = sum(branch1x1, [])

    branch3x3 = []
    branch3x3.append(Conv_block(num_filters[1][0], (1, 1), name='3x3-1_%s'%(name), **kwargs))
    branch3x3.append(Conv_block(num_filters[1][1], (3, 3), name='3x3-2_%s'%(name), **kwargs))
    branch3x3 = sum(branch3x3, [])

    branch5x5 = []
    branch5x5.append(Conv_block(num_filters[2][0], (1, 1), name='5x5-1_%s'%(name), **kwargs))
    branch5x5.append(Conv_block(num_filters[2][1], (3, 3), name='5x5-2_%s'%(name), **kwargs))
    branch5x5.append(Conv_block(num_filters[2][1], (3, 3), name='5x5-3_%s' % (name), **kwargs))
    branch5x5 = sum(branch5x5, [])

    # branch7x7 = []
    # branch7x7.append(Conv_block(64, (1, 1), name='7x7-1_%s'%(name), **kwargs))
    # branch7x7.append(Conv_block(96, (1, 7), name='7x7-2_%s'%(name), **kwargs))
    # branch7x7.append(Conv_block(96, (7, 1), name='7x7-3_%s'%(name), **kwargs))
    # branch7x7 = sum(branch7x7, [])

    branch_pool = []
    branch_pool.append([AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='aver_pool_%s'%(name))])
    branch_pool.append(Conv_block(num_filters[3], (1, 1), name='conv_pool_%s'%(name), **kwargs))
    branch_pool = sum(branch_pool, [])

    concat = Concatenate(axis=channel_axis, name='conc_%s'%(name))

    # return [branch1x1, branch3x3, branch5x5, branch7x7, branch_pool, concat]
    return [branch1x1, branch3x3, branch5x5, branch_pool, concat]

def InceptionV3_block(x): #TODO Update code
    """
    2D/3D Convolutional block followed by Activation with optional MaxPooling or Dropout.
    C-(MP)-(D)
    :param int n_filters: Number of filters used for the convolution.
    :param tuple k_size: Kernel size which is used for all three dimensions.
    :param float dropout: Adds a dropout layer if value is greater than 0.
    :param None/tuple max_pooling: Specifies if a MaxPooling layer should be added. e.g. (1,1,2) for 3D.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :param None/str kernel_reg: if L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.
    :return: x: Resulting output tensor (model).
    """
    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 3, 3)
    branch5x5 = conv2d_bn(branch5x5, 64, 3, 3)

    branch3x3dbl = conv2d_bn(x, 48, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 64, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    channel_axis = -1 if K.image_data_format() == "channels_last" else 1
    x = layers.concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

    return x

def assemble_network(inputs, layers):
    if not (isinstance(inputs, list) and isinstance(layers, list)):
        raise TypeError('passed inputs (%s) and layers (%s) need to be list.'%(type(inputs),type(layers)))

    paths = []
    subpaths = []
    parallelFlag = False
    for x_i in inputs:
        for layer in sum(layers, []):
            if type(layer) == list:
                parallelFlag = True
                x_i_temp = x_i
                for sublayer in layer:
                    x_i_temp = sublayer(x_i_temp)
                subpaths.append(x_i_temp)
            else:
                if parallelFlag == True:
                    x_i = layer(subpaths)
                    parallelFlag = False
                    subpaths = []
                else:
                    x_i = layer(x_i)
        paths.append(x_i)

    return paths

def create_shared_dcnn_network_UV():
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras import regularizers

    regu = regularizers.l2(0.01)

    # Input layers
    visible_U_1 = Input(shape=(400, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(400, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(400, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(400, 38, 1), name='V_Wire_2')

    # Define U-wire shared layers
    shared_conv_1_U = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_U', kernel_regularizer=regu)
    shared_pooling_1_U = MaxPooling2D(pool_size=(5, 3), name='Shared_2_U')
    shared_conv_2_U = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_U', kernel_regularizer=regu)
    shared_pooling_2_U = MaxPooling2D(pool_size=(3, 1), name='Shared_4_U')
    shared_conv_3_U = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_U', kernel_regularizer=regu)
    shared_pooling_3_U = MaxPooling2D(pool_size=(3, 1), name='Shared_6_U')
    shared_conv_4_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_U', kernel_regularizer=regu)
    shared_pooling_4_U = MaxPooling2D(pool_size=(3, 1), name='Shared_8_U')
    # shared_conv_5_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_9_U')

    # Define V-wire shared layers
    shared_conv_1_V = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_V', kernel_regularizer=regu)
    shared_pooling_1_V = MaxPooling2D(pool_size=(5, 3), name='Shared_2_V')
    shared_conv_2_V = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_V', kernel_regularizer=regu)
    shared_pooling_2_V = MaxPooling2D(pool_size=(3, 1), name='Shared_4_V')
    shared_conv_3_V = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_V', kernel_regularizer=regu)
    shared_pooling_3_V = MaxPooling2D(pool_size=(3, 1), name='Shared_6_V')
    shared_conv_4_V = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_V', kernel_regularizer=regu)
    shared_pooling_4_V = MaxPooling2D(pool_size=(3, 1), name='Shared_8_V')
    # shared_conv_5_V = Conv2D(128, kernel_size=3, activation='relu', name='Shared_9_V')

    # U-wire feature layers
    encoded_1_U_1 = shared_conv_1_U(visible_U_1)
    encoded_1_U_2 = shared_conv_1_U(visible_U_2)
    pooled_1_U_1 = shared_pooling_1_U(encoded_1_U_1)
    pooled_1_U_2 = shared_pooling_1_U(encoded_1_U_2)

    encoded_2_U_1 = shared_conv_2_U(pooled_1_U_1)
    encoded_2_U_2 = shared_conv_2_U(pooled_1_U_2)
    pooled_2_U_1 = shared_pooling_2_U(encoded_2_U_1)
    pooled_2_U_2 = shared_pooling_2_U(encoded_2_U_2)

    encoded_3_U_1 = shared_conv_3_U(pooled_2_U_1)
    encoded_3_U_2 = shared_conv_3_U(pooled_2_U_2)
    pooled_3_U_1 = shared_pooling_3_U(encoded_3_U_1)
    pooled_3_U_2 = shared_pooling_3_U(encoded_3_U_2)

    encoded_4_U_1 = shared_conv_4_U(pooled_3_U_1)
    encoded_4_U_2 = shared_conv_4_U(pooled_3_U_2)
    pooled_4_U_1 = shared_pooling_4_U(encoded_4_U_1)
    pooled_4_U_2 = shared_pooling_4_U(encoded_4_U_2)

    # encoded_5_U_1 = shared_conv_5_U(pooled_4_U_1)
    # encoded_5_U_2 = shared_conv_5_U(pooled_4_U_2)

    # V-wire feature layers
    encoded_1_V_1 = shared_conv_1_V(visible_V_1)
    encoded_1_V_2 = shared_conv_1_V(visible_V_2)
    pooled_1_V_1 = shared_pooling_1_V(encoded_1_V_1)
    pooled_1_V_2 = shared_pooling_1_V(encoded_1_V_2)

    encoded_2_V_1 = shared_conv_2_V(pooled_1_V_1)
    encoded_2_V_2 = shared_conv_2_V(pooled_1_V_2)
    pooled_2_V_1 = shared_pooling_2_V(encoded_2_V_1)
    pooled_2_V_2 = shared_pooling_2_V(encoded_2_V_2)

    encoded_3_V_1 = shared_conv_3_V(pooled_2_V_1)
    encoded_3_V_2 = shared_conv_3_V(pooled_2_V_2)
    pooled_3_V_1 = shared_pooling_3_V(encoded_3_V_1)
    pooled_3_V_2 = shared_pooling_3_V(encoded_3_V_2)

    encoded_4_V_1 = shared_conv_4_V(pooled_3_V_1)
    encoded_4_V_2 = shared_conv_4_V(pooled_3_V_2)
    pooled_4_V_1 = shared_pooling_4_V(encoded_4_V_1)
    pooled_4_V_2 = shared_pooling_4_V(encoded_4_V_2)

    # encoded_5_V_1 = shared_conv_5_V(pooled_4_V_1)
    # encoded_5_V_2 = shared_conv_5_V(pooled_4_V_2)

    # Merge U- and V-wire of TPC 1 and TPC 2
    merge_TPC_1 = concatenate([pooled_4_U_1, pooled_4_V_1], name='TPC_1')
    merge_TPC_2 = concatenate([pooled_4_U_2, pooled_4_V_2], name='TPC_2')

    # Flatten
    flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
    flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

    # Define shared Dense Layers
    shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

    # Dense Layers
    dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

    dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

    # Merge Dense Layers
    merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Flatten
    # flat_TPC_1_and_2 = Flatten(name='TPCs')(merge_TPC_1_2)

    # Output
    # output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    # output_xyze = Dense(2, name='Output_xyze')(merge_TPC_1_2)
    output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    #output_TPC = Dense(1, activation='sigmoid', name='Output_TPC')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])    #outputs=[output_xyze, output_TPC])

def create_shared_DEEPcnn_network():
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras import regularizers

    regu = regularizers.l2(0.01)

    # Input layers
    visible_U_1 = Input(shape=(2048, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(2048, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(2048, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(2048, 38, 1), name='V_Wire_2')

    # Define U-wire shared layers
    shared_conv_1_U = Conv2D(16, kernel_size=5, activation='relu', name='Shared_1_U', kernel_regularizer=regu)
    shared_conv_2_U = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_2_U', kernel_regularizer=regu)
    shared_pooling_1_U = MaxPooling2D(pool_size=(5, 1), name='Shared_1p_U')
    shared_conv_3_U = Conv2D(64, kernel_size=(3, 1), activation='relu', name='Shared_3_U', kernel_regularizer=regu)
    shared_conv_4_U = Conv2D(100, kernel_size=(3, 1), activation='relu', name='Shared_4_U', kernel_regularizer=regu)
    shared_pooling_2_U = MaxPooling2D(pool_size=(5, 1), name='Shared_2p_U')
    shared_conv_5_U = Conv2D(200, kernel_size=(3, 1), activation='relu', name='Shared_5_U', kernel_regularizer=regu)
    shared_conv_6_U = Conv2D(300, kernel_size=(3, 1), activation='relu', name='Shared_6_U', kernel_regularizer=regu)
    shared_pooling_3_U = MaxPooling2D(pool_size=(3, 3), name='Shared_3p_U')
    shared_conv_7_U = Conv2D(400, kernel_size=(3, 1), activation='relu', name='Shared_7_U', kernel_regularizer=regu)
    shared_conv_8_U = Conv2D(500, kernel_size=(3, 1), activation='relu', name='Shared_8_U', kernel_regularizer=regu)
    shared_pooling_4_U = MaxPooling2D(pool_size=(3, 1), name='Shared_4p_U')

    # Define V-wire shared layers
    shared_conv_1_V = Conv2D(16, kernel_size=5, activation='relu', name='Shared_1_V', kernel_regularizer=regu)
    shared_conv_2_V = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_2_V', kernel_regularizer=regu)
    shared_pooling_1_V = MaxPooling2D(pool_size=(5, 1), name='Shared_1p_V')
    shared_conv_3_V = Conv2D(64, kernel_size=(3, 1), activation='relu', name='Shared_3_V', kernel_regularizer=regu)
    shared_conv_4_V = Conv2D(100, kernel_size=(3, 1), activation='relu', name='Shared_4_V', kernel_regularizer=regu)
    shared_pooling_2_V = MaxPooling2D(pool_size=(5, 1), name='Shared_2p_V')
    shared_conv_5_V = Conv2D(200, kernel_size=(3, 1), activation='relu', name='Shared_5_V', kernel_regularizer=regu)
    shared_conv_6_V = Conv2D(300, kernel_size=(3, 1), activation='relu', name='Shared_6_V', kernel_regularizer=regu)
    shared_pooling_3_V = MaxPooling2D(pool_size=(3, 3), name='Shared_3p_V')
    shared_conv_7_V = Conv2D(400, kernel_size=(3, 1), activation='relu', name='Shared_7_V', kernel_regularizer=regu)
    shared_conv_8_V = Conv2D(500, kernel_size=(3, 1), activation='relu', name='Shared_8_V', kernel_regularizer=regu)
    shared_pooling_4_V = MaxPooling2D(pool_size=(3, 1), name='Shared_4p_V')

    # U-wire feature layers
    encoded_1_U_1 = shared_conv_1_U(visible_U_1)
    encoded_1_U_2 = shared_conv_1_U(visible_U_2)

    encoded_2_U_1 = shared_conv_2_U(encoded_1_U_1)
    encoded_2_U_2 = shared_conv_2_U(encoded_1_U_2)

    pooled_1_U_1 = shared_pooling_1_U(encoded_2_U_1)
    pooled_1_U_2 = shared_pooling_1_U(encoded_2_U_2)

    encoded_3_U_1 = shared_conv_3_U(pooled_1_U_1)
    encoded_3_U_2 = shared_conv_3_U(pooled_1_U_2)

    encoded_4_U_1 = shared_conv_4_U(encoded_3_U_1)
    encoded_4_U_2 = shared_conv_4_U(encoded_3_U_2)

    pooled_2_U_1 = shared_pooling_2_U(encoded_4_U_1)
    pooled_2_U_2 = shared_pooling_2_U(encoded_4_U_2)

    encoded_5_U_1 = shared_conv_5_U(pooled_2_U_1)
    encoded_5_U_2 = shared_conv_5_U(pooled_2_U_2)

    encoded_6_U_1 = shared_conv_6_U(encoded_5_U_1)
    encoded_6_U_2 = shared_conv_6_U(encoded_5_U_2)

    pooled_3_U_1 = shared_pooling_3_U(encoded_6_U_1)
    pooled_3_U_2 = shared_pooling_3_U(encoded_6_U_2)

    encoded_7_U_1 = shared_conv_7_U(pooled_3_U_1)
    encoded_7_U_2 = shared_conv_7_U(pooled_3_U_2)

    encoded_8_U_1 = shared_conv_8_U(encoded_7_U_1)
    encoded_8_U_2 = shared_conv_8_U(encoded_7_U_2)

    pooled_4_U_1 = shared_pooling_4_U(encoded_8_U_1)
    pooled_4_U_2 = shared_pooling_4_U(encoded_8_U_2)


    # V-wire feature layers
    encoded_1_V_1 = shared_conv_1_V(visible_V_1)
    encoded_1_V_2 = shared_conv_1_V(visible_V_2)

    encoded_2_V_1 = shared_conv_2_V(encoded_1_V_1)
    encoded_2_V_2 = shared_conv_2_V(encoded_1_V_2)

    pooled_1_V_1 = shared_pooling_1_V(encoded_2_V_1)
    pooled_1_V_2 = shared_pooling_1_V(encoded_2_V_2)

    encoded_3_V_1 = shared_conv_3_V(pooled_1_V_1)
    encoded_3_V_2 = shared_conv_3_V(pooled_1_V_2)

    encoded_4_V_1 = shared_conv_4_V(encoded_3_V_1)
    encoded_4_V_2 = shared_conv_4_V(encoded_3_V_2)

    pooled_2_V_1 = shared_pooling_2_V(encoded_4_V_1)
    pooled_2_V_2 = shared_pooling_2_V(encoded_4_V_2)

    encoded_5_V_1 = shared_conv_5_V(pooled_2_V_1)
    encoded_5_V_2 = shared_conv_5_V(pooled_2_V_2)

    encoded_6_V_1 = shared_conv_6_V(encoded_5_V_1)
    encoded_6_V_2 = shared_conv_6_V(encoded_5_V_2)

    pooled_3_V_1 = shared_pooling_3_V(encoded_6_V_1)
    pooled_3_V_2 = shared_pooling_3_V(encoded_6_V_2)

    encoded_7_V_1 = shared_conv_7_V(pooled_3_V_1)
    encoded_7_V_2 = shared_conv_7_V(pooled_3_V_2)

    encoded_8_V_1 = shared_conv_8_V(encoded_7_V_1)
    encoded_8_V_2 = shared_conv_8_V(encoded_7_V_2)

    pooled_4_V_1 = shared_pooling_4_V(encoded_8_V_1)
    pooled_4_V_2 = shared_pooling_4_V(encoded_8_V_2)

    # Merge U- and V-wire of TPC 1 and TPC 2
    merge_TPC_1 = concatenate([pooled_4_U_1, pooled_4_V_1], name='TPC_1')
    merge_TPC_2 = concatenate([pooled_4_U_2, pooled_4_V_2], name='TPC_2')

    # Flatten
    flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
    flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

    # Define shared Dense Layers
    shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

    # Dense Layers
    dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

    dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

    # Merge Dense Layers
    merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Flatten
    # flat_TPC_1_and_2 = Flatten(name='TPCs')(merge_TPC_1_2)

    # Output
    # output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    # output_xyze = Dense(2, name='Output_xyze')(merge_TPC_1_2)
    output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    #output_TPC = Dense(1, activation='sigmoid', name='Output_TPC')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])    #outputs=[output_xyze, output_TPC])

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

