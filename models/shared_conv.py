def create_shared_dcnn_network_U():
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import Concatenate, Add
    from keras import regularizers

    regu = None #regularizers.l2(1.e-2)
    init = "glorot_uniform"
    act = "relu"
    padding = "same"

    # Input layers
    visible_1 = Input(shape=(350, 38, 1), name='Wire_1')
    visible_2 = Input(shape=(350, 38, 1), name='Wire_2')

    # Define U-wire shared layers
    shared_conv_1 = Conv2D(16, kernel_size=(5, 3), name='Shared_1', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
    shared_conv_2 = Conv2D(32, kernel_size=(5, 3), name='Shared_2', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
    shared_pooling_1 = MaxPooling2D(pool_size=(4, 2), name='Shared_3', padding=padding)
    shared_conv_3 = Conv2D(64, kernel_size=(3, 3), name='Shared_4', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
    shared_conv_4 = Conv2D(128, kernel_size=(3, 3), name='Shared_5', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
    shared_pooling_2 = MaxPooling2D(pool_size=(4, 2), name='Shared_6', padding=padding)
    shared_conv_5 = Conv2D(256, kernel_size=(3, 3), name='Shared_7', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
    shared_pooling_3 = MaxPooling2D(pool_size=(2, 2), name='Shared_8', padding=padding)

    # U-wire feature layers
    encoded_1_1 = shared_conv_1(visible_1)
    encoded_1_2 = shared_conv_1(visible_2)
    encoded_2_1 = shared_conv_2(encoded_1_1)
    encoded_2_2 = shared_conv_2(encoded_1_2)
    pooled_1_1 = shared_pooling_1(encoded_2_1)
    pooled_1_2 = shared_pooling_1(encoded_2_2)

    encoded_3_1 = shared_conv_3(pooled_1_1)
    encoded_3_2 = shared_conv_3(pooled_1_2)
    encoded_4_1 = shared_conv_4(encoded_3_1)
    encoded_4_2 = shared_conv_4(encoded_3_2)
    pooled_2_1 = shared_pooling_2(encoded_4_1)
    pooled_2_2 = shared_pooling_2(encoded_4_2)

    encoded_5_1 = shared_conv_5(pooled_2_1)
    encoded_5_2 = shared_conv_5(pooled_2_2)
    pooled_3_1 = shared_pooling_3(encoded_5_1)
    pooled_3_2 = shared_pooling_3(encoded_5_2)

    shared_flat = Flatten(name='flat1')

    # Flatten
    flat_1 = shared_flat(pooled_3_1)
    flat_2 = shared_flat(pooled_3_2)

    # Define shared Dense Layers
    shared_dense_1 = Dense(32, name='Shared_1_Dense', activation=act, kernel_initializer=init, kernel_regularizer=regu)
    shared_dense_2 = Dense(8, name='Shared_2_Dense', activation=act, kernel_initializer=init, kernel_regularizer=regu)

    # Dense Layers
    dense_1_1 = shared_dense_1(flat_1)
    dense_1_2 = shared_dense_1(flat_2)

    dense_2_1 = shared_dense_2(dense_1_1)
    dense_2_2 = shared_dense_2(dense_1_2)

    # Merge Dense Layers
    merge_1_2 = Concatenate(name='Flat_1_and_2')([dense_2_1, dense_2_2])

    # Output
    output = Dense(2, name='Output', activation='softmax', kernel_initializer=init)(merge_1_2)

    return Model(inputs=[visible_1, visible_2], outputs=[output])

def create_shared_dcnn_network_UV():
    from keras.utils import plot_model
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
    from keras.utils import plot_model
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

def create_inception_network():
    from keras.utils import plot_model
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

    # shared Layer U-wire
    conv_1_U = Conv2D(64, (7, 7), activation='relu')

    # shared Inception module U-wire
    shared_inception_1_a_U = Conv2D(64, (1, 1), padding='same', activation='relu')
    shared_inception_1_b1_U = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_1_b2_U = Conv2D(128, (3, 3), padding='same', activation='relu')
    shared_inception_1_c1_U = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_1_c2_U = Conv2D(32, (5, 5), padding='same', activation='relu')
    shared_inception_1_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_1_d2_U = Conv2D(32, (1, 1), padding='same', activation='relu')

    shared_inception_2_a_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b1_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b2_U = Conv2D(192, (3, 3), padding='same', activation='relu')
    shared_inception_2_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_2_c2_U = Conv2D(96, (5, 5), padding='same', activation='relu')
    shared_inception_2_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_2_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_3_a_U = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_3_b1_U = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_3_b2_U = Conv2D(208, (3, 3), padding='same', activation='relu')
    shared_inception_3_c1_U = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_3_c2_U = Conv2D(48, (5, 5), padding='same', activation='relu')
    shared_inception_3_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_3_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_4_a_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_4_b1_U = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_4_b2_U = Conv2D(224, (3, 3), padding='same', activation='relu')
    shared_inception_4_c1_U = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_4_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_4_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_4_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_5_a_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b1_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b2_U = Conv2D(256, (3, 3), padding='same', activation='relu')
    shared_inception_5_c1_U = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_5_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_5_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_5_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_6_a_U = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_6_b1_U = Conv2D(144, (1, 1), padding='same', activation='relu')
    shared_inception_6_b2_U = Conv2D(288, (3, 3), padding='same', activation='relu')
    shared_inception_6_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_6_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_6_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_6_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_7_a_U = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_7_b1_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_7_b2_U = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_7_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_7_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_7_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_7_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_8_a_U = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_8_b1_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_8_b2_U = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_8_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_8_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_8_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_8_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_9_a_U = Conv2D(384, (1, 1), padding='same', activation='relu')
    shared_inception_9_b1_U = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_9_b2_U = Conv2D(384, (3, 3), padding='same', activation='relu')
    shared_inception_9_c1_U = Conv2D(48, (1, 1), padding='same', activation='relu')
    shared_inception_9_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_9_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_9_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    # shared Layer V-wire
    conv_1_V = Conv2D(64, (7, 7), activation='relu')

    # shared Inception module V-wire
    shared_inception_1_a_V = Conv2D(64, (1, 1), padding='same', activation='relu')
    shared_inception_1_b1_V = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_1_b2_V = Conv2D(128, (3, 3), padding='same', activation='relu')
    shared_inception_1_c1_V = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_1_c2_V = Conv2D(32, (5, 5), padding='same', activation='relu')
    shared_inception_1_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_1_d2_V = Conv2D(32, (1, 1), padding='same', activation='relu')

    shared_inception_2_a_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b1_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b2_V = Conv2D(192, (3, 3), padding='same', activation='relu')
    shared_inception_2_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_2_c2_V = Conv2D(96, (5, 5), padding='same', activation='relu')
    shared_inception_2_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_2_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_3_a_V = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_3_b1_V = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_3_b2_V = Conv2D(208, (3, 3), padding='same', activation='relu')
    shared_inception_3_c1_V = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_3_c2_V = Conv2D(48, (5, 5), padding='same', activation='relu')
    shared_inception_3_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_3_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_4_a_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_4_b1_V = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_4_b2_V = Conv2D(224, (3, 3), padding='same', activation='relu')
    shared_inception_4_c1_V = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_4_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_4_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_4_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_5_a_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b1_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b2_V = Conv2D(256, (3, 3), padding='same', activation='relu')
    shared_inception_5_c1_V = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_5_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_5_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_5_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_6_a_V = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_6_b1_V = Conv2D(144, (1, 1), padding='same', activation='relu')
    shared_inception_6_b2_V = Conv2D(288, (3, 3), padding='same', activation='relu')
    shared_inception_6_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_6_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_6_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_6_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_7_a_V = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_7_b1_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_7_b2_V = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_7_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_7_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_7_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_7_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_8_a_V = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_8_b1_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_8_b2_V = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_8_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_8_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_8_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_8_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_9_a_V = Conv2D(384, (1, 1), padding='same', activation='relu')
    shared_inception_9_b1_V = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_9_b2_V = Conv2D(384, (3, 3), padding='same', activation='relu')
    shared_inception_9_c1_V = Conv2D(48, (1, 1), padding='same', activation='relu')
    shared_inception_9_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_9_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_9_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    # Application U-1
    # U_1_conv_1 = conv_1_U(visible_U_1)
    # U_1_conv_1 = MaxPooling2D(pool_size=3)(U_1_conv_1)

    U_1_1_a = shared_inception_1_a_U(visible_U_1)
    U_1_1_b = shared_inception_1_b2_U(shared_inception_1_b1_U(visible_U_1))
    U_1_1_c = shared_inception_1_c2_U(shared_inception_1_c1_U(visible_U_1))
    U_1_1_d = shared_inception_1_d2_U(shared_inception_1_d1_U(visible_U_1))
    inception_1_U_1 = concatenate([U_1_1_a, U_1_1_b, U_1_1_c, U_1_1_d], axis=3)

    U_1_2_a = shared_inception_2_a_U(inception_1_U_1)
    U_1_2_b = shared_inception_2_b2_U(shared_inception_2_b1_U(inception_1_U_1))
    U_1_2_c = shared_inception_2_c2_U(shared_inception_2_c1_U(inception_1_U_1))
    U_1_2_d = shared_inception_2_d2_U(shared_inception_2_d1_U(inception_1_U_1))
    inception_2_U_1 = concatenate([U_1_2_a, U_1_2_b, U_1_2_c, U_1_2_d], axis=3)

    inception_2_U_1 = MaxPooling2D(pool_size=3)(inception_2_U_1)

    U_1_3_a = shared_inception_3_a_U(inception_2_U_1)
    U_1_3_b = shared_inception_3_b2_U(shared_inception_3_b1_U(inception_2_U_1))
    U_1_3_c = shared_inception_3_c2_U(shared_inception_3_c1_U(inception_2_U_1))
    U_1_3_d = shared_inception_3_d2_U(shared_inception_3_d1_U(inception_2_U_1))
    inception_3_U_1 = concatenate([U_1_3_a, U_1_3_b, U_1_3_c, U_1_3_d], axis=3)

    U_1_4_a = shared_inception_4_a_U(inception_3_U_1)
    U_1_4_b = shared_inception_4_b2_U(shared_inception_4_b1_U(inception_3_U_1))
    U_1_4_c = shared_inception_4_c2_U(shared_inception_4_c1_U(inception_3_U_1))
    U_1_4_d = shared_inception_4_d2_U(shared_inception_4_d1_U(inception_3_U_1))
    inception_4_U_1 = concatenate([U_1_4_a, U_1_4_b, U_1_4_c, U_1_4_d], axis=3)

    inception_4_U_1 = MaxPooling2D(pool_size=3)(inception_4_U_1)

    U_1_5_a = shared_inception_5_a_U(inception_4_U_1)
    U_1_5_b = shared_inception_5_b2_U(shared_inception_5_b1_U(inception_4_U_1))
    U_1_5_c = shared_inception_5_c2_U(shared_inception_5_c1_U(inception_4_U_1))
    U_1_5_d = shared_inception_5_d2_U(shared_inception_5_d1_U(inception_4_U_1))
    inception_5_U_1 = concatenate([U_1_5_a, U_1_5_b, U_1_5_c, U_1_5_d], axis=3)

    U_1_6_a = shared_inception_6_a_U(inception_5_U_1)
    U_1_6_b = shared_inception_6_b2_U(shared_inception_6_b1_U(inception_5_U_1))
    U_1_6_c = shared_inception_6_c2_U(shared_inception_6_c1_U(inception_5_U_1))
    U_1_6_d = shared_inception_6_d2_U(shared_inception_6_d1_U(inception_5_U_1))
    inception_6_U_1 = concatenate([U_1_6_a, U_1_6_b, U_1_6_c, U_1_6_d], axis=3)

    U_1_7_a = shared_inception_7_a_U(inception_6_U_1)
    U_1_7_b = shared_inception_7_b2_U(shared_inception_7_b1_U(inception_6_U_1))
    U_1_7_c = shared_inception_7_c2_U(shared_inception_7_c1_U(inception_6_U_1))
    U_1_7_d = shared_inception_7_d2_U(shared_inception_7_d1_U(inception_6_U_1))
    inception_7_U_1 = concatenate([U_1_7_a, U_1_7_b, U_1_7_c, U_1_7_d], axis=3)

    inception_7_U_1 = MaxPooling2D(pool_size=3)(inception_7_U_1)

    U_1_8_a = shared_inception_8_a_U(inception_7_U_1)
    U_1_8_b = shared_inception_8_b2_U(shared_inception_8_b1_U(inception_7_U_1))
    U_1_8_c = shared_inception_8_c2_U(shared_inception_8_c1_U(inception_7_U_1))
    U_1_8_d = shared_inception_8_d2_U(shared_inception_8_d1_U(inception_7_U_1))
    inception_8_U_1 = concatenate([U_1_8_a, U_1_8_b, U_1_8_c, U_1_8_d], axis=3)

    U_1_9_a = shared_inception_9_a_U(inception_8_U_1)
    U_1_9_b = shared_inception_9_b2_U(shared_inception_9_b1_U(inception_8_U_1))
    U_1_9_c = shared_inception_9_c2_U(shared_inception_9_c1_U(inception_8_U_1))
    U_1_9_d = shared_inception_9_d2_U(shared_inception_9_d1_U(inception_8_U_1))
    inception_9_U_1 = concatenate([U_1_9_a, U_1_9_b, U_1_9_c, U_1_9_d], axis=3)

    inception_9_U_1 = MaxPooling2D(pool_size=(7, 1))(inception_9_U_1)


    # Application U-2
    # U_2_conv_1 = conv_1_U(visible_U_2)
    # U_2_conv_1 = MaxPooling2D(pool_size=3)(U_2_conv_1)

    U_2_1_a = shared_inception_1_a_U(visible_U_2)
    U_2_1_b = shared_inception_1_b2_U(shared_inception_1_b1_U(visible_U_2))
    U_2_1_c = shared_inception_1_c2_U(shared_inception_1_c1_U(visible_U_2))
    U_2_1_d = shared_inception_1_d2_U(shared_inception_1_d1_U(visible_U_2))
    inception_1_U_2 = concatenate([U_2_1_a, U_2_1_b, U_2_1_c, U_2_1_d], axis=3)

    U_2_2_a = shared_inception_2_a_U(inception_1_U_2)
    U_2_2_b = shared_inception_2_b2_U(shared_inception_2_b1_U(inception_1_U_2))
    U_2_2_c = shared_inception_2_c2_U(shared_inception_2_c1_U(inception_1_U_2))
    U_2_2_d = shared_inception_2_d2_U(shared_inception_2_d1_U(inception_1_U_2))
    inception_2_U_2 = concatenate([U_2_2_a, U_2_2_b, U_2_2_c, U_2_2_d], axis=3)

    inception_2_U_2 = MaxPooling2D(pool_size=3)(inception_2_U_2)

    U_2_3_a = shared_inception_3_a_U(inception_2_U_2)
    U_2_3_b = shared_inception_3_b2_U(shared_inception_3_b1_U(inception_2_U_2))
    U_2_3_c = shared_inception_3_c2_U(shared_inception_3_c1_U(inception_2_U_2))
    U_2_3_d = shared_inception_3_d2_U(shared_inception_3_d1_U(inception_2_U_2))
    inception_3_U_2 = concatenate([U_2_3_a, U_2_3_b, U_2_3_c, U_2_3_d], axis=3)

    U_2_4_a = shared_inception_4_a_U(inception_3_U_2)
    U_2_4_b = shared_inception_4_b2_U(shared_inception_4_b1_U(inception_3_U_2))
    U_2_4_c = shared_inception_4_c2_U(shared_inception_4_c1_U(inception_3_U_2))
    U_2_4_d = shared_inception_4_d2_U(shared_inception_4_d1_U(inception_3_U_2))
    inception_4_U_2 = concatenate([U_2_4_a, U_2_4_b, U_2_4_c, U_2_4_d], axis=3)

    inception_4_U_2 = MaxPooling2D(pool_size=3)(inception_4_U_2)

    U_2_5_a = shared_inception_5_a_U(inception_4_U_2)
    U_2_5_b = shared_inception_5_b2_U(shared_inception_5_b1_U(inception_4_U_2))
    U_2_5_c = shared_inception_5_c2_U(shared_inception_5_c1_U(inception_4_U_2))
    U_2_5_d = shared_inception_5_d2_U(shared_inception_5_d1_U(inception_4_U_2))
    inception_5_U_2 = concatenate([U_2_5_a, U_2_5_b, U_2_5_c, U_2_5_d], axis=3)

    U_2_6_a = shared_inception_6_a_U(inception_5_U_2)
    U_2_6_b = shared_inception_6_b2_U(shared_inception_6_b1_U(inception_5_U_2))
    U_2_6_c = shared_inception_6_c2_U(shared_inception_6_c1_U(inception_5_U_2))
    U_2_6_d = shared_inception_6_d2_U(shared_inception_6_d1_U(inception_5_U_2))
    inception_6_U_2 = concatenate([U_2_6_a, U_2_6_b, U_2_6_c, U_2_6_d], axis=3)

    U_2_7_a = shared_inception_7_a_U(inception_6_U_2)
    U_2_7_b = shared_inception_7_b2_U(shared_inception_7_b1_U(inception_6_U_2))
    U_2_7_c = shared_inception_7_c2_U(shared_inception_7_c1_U(inception_6_U_2))
    U_2_7_d = shared_inception_7_d2_U(shared_inception_7_d1_U(inception_6_U_2))
    inception_7_U_2 = concatenate([U_2_7_a, U_2_7_b, U_2_7_c, U_2_7_d], axis=3)

    inception_7_U_2 = MaxPooling2D(pool_size=3)(inception_7_U_2)

    U_2_8_a = shared_inception_8_a_U(inception_7_U_2)
    U_2_8_b = shared_inception_8_b2_U(shared_inception_8_b1_U(inception_7_U_2))
    U_2_8_c = shared_inception_8_c2_U(shared_inception_8_c1_U(inception_7_U_2))
    U_2_8_d = shared_inception_8_d2_U(shared_inception_8_d1_U(inception_7_U_2))
    inception_8_U_2 = concatenate([U_2_8_a, U_2_8_b, U_2_8_c, U_2_8_d], axis=3)

    U_2_9_a = shared_inception_9_a_U(inception_8_U_2)
    U_2_9_b = shared_inception_9_b2_U(shared_inception_9_b1_U(inception_8_U_2))
    U_2_9_c = shared_inception_9_c2_U(shared_inception_9_c1_U(inception_8_U_2))
    U_2_9_d = shared_inception_9_d2_U(shared_inception_9_d1_U(inception_8_U_2))
    inception_9_U_2 = concatenate([U_2_9_a, U_2_9_b, U_2_9_c, U_2_9_d], axis=3)

    inception_9_U_2 = MaxPooling2D(pool_size=(7, 1))(inception_9_U_2)



    # Application V-1
    # V_1_conv_1 = conv_1_V(visible_V_1)
    # V_1_conv_1 = MaxPooling2D(pool_size=3)(V_1_conv_1)

    V_1_1_a = shared_inception_1_a_V(visible_V_1)
    V_1_1_b = shared_inception_1_b2_V(shared_inception_1_b1_V(visible_V_1))
    V_1_1_c = shared_inception_1_c2_V(shared_inception_1_c1_V(visible_V_1))
    V_1_1_d = shared_inception_1_d2_V(shared_inception_1_d1_V(visible_V_1))
    inception_1_V_1 = concatenate([V_1_1_a, V_1_1_b, V_1_1_c, V_1_1_d], axis=3)

    V_1_2_a = shared_inception_2_a_V(inception_1_V_1)
    V_1_2_b = shared_inception_2_b2_V(shared_inception_2_b1_V(inception_1_V_1))
    V_1_2_c = shared_inception_2_c2_V(shared_inception_2_c1_V(inception_1_V_1))
    V_1_2_d = shared_inception_2_d2_V(shared_inception_2_d1_V(inception_1_V_1))
    inception_2_V_1 = concatenate([V_1_2_a, V_1_2_b, V_1_2_c, V_1_2_d], axis=3)

    inception_2_V_1 = MaxPooling2D(pool_size=3)(inception_2_V_1)

    V_1_3_a = shared_inception_3_a_V(inception_2_V_1)
    V_1_3_b = shared_inception_3_b2_V(shared_inception_3_b1_V(inception_2_V_1))
    V_1_3_c = shared_inception_3_c2_V(shared_inception_3_c1_V(inception_2_V_1))
    V_1_3_d = shared_inception_3_d2_V(shared_inception_3_d1_V(inception_2_V_1))
    inception_3_V_1 = concatenate([V_1_3_a, V_1_3_b, V_1_3_c, V_1_3_d], axis=3)

    V_1_4_a = shared_inception_4_a_V(inception_3_V_1)
    V_1_4_b = shared_inception_4_b2_V(shared_inception_4_b1_V(inception_3_V_1))
    V_1_4_c = shared_inception_4_c2_V(shared_inception_4_c1_V(inception_3_V_1))
    V_1_4_d = shared_inception_4_d2_V(shared_inception_4_d1_V(inception_3_V_1))
    inception_4_V_1 = concatenate([V_1_4_a, V_1_4_b, V_1_4_c, V_1_4_d], axis=3)

    inception_4_V_1 = MaxPooling2D(pool_size=3)(inception_4_V_1)

    V_1_5_a = shared_inception_5_a_V(inception_4_V_1)
    V_1_5_b = shared_inception_5_b2_V(shared_inception_5_b1_V(inception_4_V_1))
    V_1_5_c = shared_inception_5_c2_V(shared_inception_5_c1_V(inception_4_V_1))
    V_1_5_d = shared_inception_5_d2_V(shared_inception_5_d1_V(inception_4_V_1))
    inception_5_V_1 = concatenate([V_1_5_a, V_1_5_b, V_1_5_c, V_1_5_d], axis=3)

    V_1_6_a = shared_inception_6_a_V(inception_5_V_1)
    V_1_6_b = shared_inception_6_b2_V(shared_inception_6_b1_V(inception_5_V_1))
    V_1_6_c = shared_inception_6_c2_V(shared_inception_6_c1_V(inception_5_V_1))
    V_1_6_d = shared_inception_6_d2_V(shared_inception_6_d1_V(inception_5_V_1))
    inception_6_V_1 = concatenate([V_1_6_a, V_1_6_b, V_1_6_c, V_1_6_d], axis=3)

    V_1_7_a = shared_inception_7_a_V(inception_6_V_1)
    V_1_7_b = shared_inception_7_b2_V(shared_inception_7_b1_V(inception_6_V_1))
    V_1_7_c = shared_inception_7_c2_V(shared_inception_7_c1_V(inception_6_V_1))
    V_1_7_d = shared_inception_7_d2_V(shared_inception_7_d1_V(inception_6_V_1))
    inception_7_V_1 = concatenate([V_1_7_a, V_1_7_b, V_1_7_c, V_1_7_d], axis=3)

    inception_7_V_1 = MaxPooling2D(pool_size=3)(inception_7_V_1)

    V_1_8_a = shared_inception_8_a_V(inception_7_V_1)
    V_1_8_b = shared_inception_8_b2_V(shared_inception_8_b1_V(inception_7_V_1))
    V_1_8_c = shared_inception_8_c2_V(shared_inception_8_c1_V(inception_7_V_1))
    V_1_8_d = shared_inception_8_d2_V(shared_inception_8_d1_V(inception_7_V_1))
    inception_8_V_1 = concatenate([V_1_8_a, V_1_8_b, V_1_8_c, V_1_8_d], axis=3)

    V_1_9_a = shared_inception_9_a_V(inception_8_V_1)
    V_1_9_b = shared_inception_9_b2_V(shared_inception_9_b1_V(inception_8_V_1))
    V_1_9_c = shared_inception_9_c2_V(shared_inception_9_c1_V(inception_8_V_1))
    V_1_9_d = shared_inception_9_d2_V(shared_inception_9_d1_V(inception_8_V_1))
    inception_9_V_1 = concatenate([V_1_9_a, V_1_9_b, V_1_9_c, V_1_9_d], axis=3)

    inception_9_V_1 = MaxPooling2D(pool_size=(7, 1))(inception_9_V_1)


    # Application V-2
    # V_2_conv_1 = conv_1_V(visible_V_2)
    # V_2_conv_1 = MaxPooling2D(pool_size=3)(V_2_conv_1)

    V_2_1_a = shared_inception_1_a_V(visible_V_2)
    V_2_1_b = shared_inception_1_b2_V(shared_inception_1_b1_V(visible_V_2))
    V_2_1_c = shared_inception_1_c2_V(shared_inception_1_c1_V(visible_V_2))
    V_2_1_d = shared_inception_1_d2_V(shared_inception_1_d1_V(visible_V_2))
    inception_1_V_2 = concatenate([V_2_1_a, V_2_1_b, V_2_1_c, V_2_1_d], axis=3)

    V_2_2_a = shared_inception_2_a_V(inception_1_V_2)
    V_2_2_b = shared_inception_2_b2_V(shared_inception_2_b1_V(inception_1_V_2))
    V_2_2_c = shared_inception_2_c2_V(shared_inception_2_c1_V(inception_1_V_2))
    V_2_2_d = shared_inception_2_d2_V(shared_inception_2_d1_V(inception_1_V_2))
    inception_2_V_2 = concatenate([V_2_2_a, V_2_2_b, V_2_2_c, V_2_2_d], axis=3)

    inception_2_V_2 = MaxPooling2D(pool_size=3)(inception_2_V_2)

    V_2_3_a = shared_inception_3_a_V(inception_2_V_2)
    V_2_3_b = shared_inception_3_b2_V(shared_inception_3_b1_V(inception_2_V_2))
    V_2_3_c = shared_inception_3_c2_V(shared_inception_3_c1_V(inception_2_V_2))
    V_2_3_d = shared_inception_3_d2_V(shared_inception_3_d1_V(inception_2_V_2))
    inception_3_V_2 = concatenate([V_2_3_a, V_2_3_b, V_2_3_c, V_2_3_d], axis=3)

    V_2_4_a = shared_inception_4_a_V(inception_3_V_2)
    V_2_4_b = shared_inception_4_b2_V(shared_inception_4_b1_V(inception_3_V_2))
    V_2_4_c = shared_inception_4_c2_V(shared_inception_4_c1_V(inception_3_V_2))
    V_2_4_d = shared_inception_4_d2_V(shared_inception_4_d1_V(inception_3_V_2))
    inception_4_V_2 = concatenate([V_2_4_a, V_2_4_b, V_2_4_c, V_2_4_d], axis=3)

    inception_4_V_2 = MaxPooling2D(pool_size=3)(inception_4_V_2)

    V_2_5_a = shared_inception_5_a_V(inception_4_V_2)
    V_2_5_b = shared_inception_5_b2_V(shared_inception_5_b1_V(inception_4_V_2))
    V_2_5_c = shared_inception_5_c2_V(shared_inception_5_c1_V(inception_4_V_2))
    V_2_5_d = shared_inception_5_d2_V(shared_inception_5_d1_V(inception_4_V_2))
    inception_5_V_2 = concatenate([V_2_5_a, V_2_5_b, V_2_5_c, V_2_5_d], axis=3)

    V_2_6_a = shared_inception_6_a_V(inception_5_V_2)
    V_2_6_b = shared_inception_6_b2_V(shared_inception_6_b1_V(inception_5_V_2))
    V_2_6_c = shared_inception_6_c2_V(shared_inception_6_c1_V(inception_5_V_2))
    V_2_6_d = shared_inception_6_d2_V(shared_inception_6_d1_V(inception_5_V_2))
    inception_6_V_2 = concatenate([V_2_6_a, V_2_6_b, V_2_6_c, V_2_6_d], axis=3)

    V_2_7_a = shared_inception_7_a_V(inception_6_V_2)
    V_2_7_b = shared_inception_7_b2_V(shared_inception_7_b1_V(inception_6_V_2))
    V_2_7_c = shared_inception_7_c2_V(shared_inception_7_c1_V(inception_6_V_2))
    V_2_7_d = shared_inception_7_d2_V(shared_inception_7_d1_V(inception_6_V_2))
    inception_7_V_2 = concatenate([V_2_7_a, V_2_7_b, V_2_7_c, V_2_7_d], axis=3)

    inception_7_V_2 = MaxPooling2D(pool_size=3)(inception_7_V_2)

    V_2_8_a = shared_inception_8_a_V(inception_7_V_2)
    V_2_8_b = shared_inception_8_b2_V(shared_inception_8_b1_V(inception_7_V_2))
    V_2_8_c = shared_inception_8_c2_V(shared_inception_8_c1_V(inception_7_V_2))
    V_2_8_d = shared_inception_8_d2_V(shared_inception_8_d1_V(inception_7_V_2))
    inception_8_V_2 = concatenate([V_2_8_a, V_2_8_b, V_2_8_c, V_2_8_d], axis=3)

    V_2_9_a = shared_inception_9_a_V(inception_8_V_2)
    V_2_9_b = shared_inception_9_b2_V(shared_inception_9_b1_V(inception_8_V_2))
    V_2_9_c = shared_inception_9_c2_V(shared_inception_9_c1_V(inception_8_V_2))
    V_2_9_d = shared_inception_9_d2_V(shared_inception_9_d1_V(inception_8_V_2))
    inception_9_V_2 = concatenate([V_2_9_a, V_2_9_b, V_2_9_c, V_2_9_d], axis=3)

    inception_9_V_2 = MaxPooling2D(pool_size=(7, 1))(inception_9_V_2)


    # Merge U- and V-wire of TPC 1 and TPC 2
    merge_TPC_1 = concatenate([inception_9_U_1, inception_9_V_1], name='TPC_1')
    merge_TPC_2 = concatenate([inception_9_U_2, inception_9_V_2], name='TPC_2')

    # Flatten
    flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
    flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

    # Define shared Dense Layers
    shared_dense_1 = Dense(16, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    shared_dense_2 = Dense(8, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

    # Dense Layers
    dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

    dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

    # Merge Dense Layers
    merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Output
    output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])


