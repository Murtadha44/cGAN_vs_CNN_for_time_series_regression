from tensorflow.keras.initializers import he_normal, he_uniform, RandomNormal, RandomUniform
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense, LeakyReLU, concatenate, Dropout, Conv1D, MaxPooling1D,\
    GlobalAveragePooling1D, Activation, BatchNormalization, Lambda, ReLU, ELU
from tensorflow.keras.activations import linear, relu, elu, selu, tanh, sigmoid
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def conv_block(num_filt, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size):
    x = Conv1D(num_filt, filters_width, padding='same', kernel_initializer='normal',
               kernel_regularizer=l2(lambda_loss_amount),use_bias=False)(x_inp)
    x = BatchNormalization(momentum=0.5)(x)
    x = act_fun(x)
    x = Conv1D(num_filt, filters_width, padding='same', kernel_initializer='normal',
               kernel_regularizer=l2(lambda_loss_amount), use_bias=False)(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = act_fun(x)
    x = MaxPooling1D(pooling_size)(x)
    return x

def conv_layers(num_conv, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size):
    x = conv_block(32, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size)
    for block_i in range(int((num_conv - 3) / 2)):
        x = conv_block(64, x, act_fun, filters_width, lambda_loss_amount, pooling_size)
    x = Conv1D(32, filters_width, padding='same', kernel_initializer='normal', kernel_regularizer=l2(lambda_loss_amount),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = act_fun(x)
    x = GlobalAveragePooling1D()(x)
    return x

def conv_block_noNorm(num_filt, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size):
    x = Conv1D(num_filt, filters_width, padding='same', kernel_initializer='normal',
               kernel_regularizer=l2(lambda_loss_amount),use_bias=True)(x_inp)
    x = act_fun(x)
    x = Conv1D(num_filt, filters_width, padding='same', kernel_initializer='normal',
               kernel_regularizer=l2(lambda_loss_amount), use_bias=True)(x)
    x = act_fun(x)
    x = MaxPooling1D(pooling_size)(x)
    return x

def conv_layers_noNorm(num_conv, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size):
    x = conv_block_noNorm(32, x_inp, act_fun, filters_width, lambda_loss_amount, pooling_size)
    for block_i in range(int((num_conv - 3) / 2)):
        x = conv_block_noNorm(64, x, act_fun, filters_width, lambda_loss_amount, pooling_size)
    x = Conv1D(32, filters_width, padding='same', kernel_initializer='normal', kernel_regularizer=l2(lambda_loss_amount),
               use_bias=True)(x)
    x = act_fun(x)
    x = GlobalAveragePooling1D()(x)
    return x



def build_cnn_fun(num_conv_layers, window_length, numChannels, filters_width, pooling_size,
                  dropout_Prob, lambda_loss_amount):
    x_all = Input(shape=(window_length, numChannels,))
    x_concat = conv_layers_noNorm(num_conv_layers, x_all, ReLU(), filters_width, lambda_loss_amount, pooling_size)

    x_concat = Dropout(dropout_Prob)(x_concat)
    x_concat = Dense(128, kernel_initializer='normal', kernel_regularizer=l2(lambda_loss_amount),
                     use_bias=True)(x_concat)
    x_concat = Activation('relu')(x_concat)
    x_concat = Dropout(dropout_Prob)(x_concat)
    out = Dense(1, kernel_initializer='normal', kernel_regularizer=l2(lambda_loss_amount),
                activation='linear')(x_concat)
    model_m = Model(inputs=x_all, outputs=out)
    return model_m



def build_generator(network):
    seed = network["seed"]
    random_normal = RandomNormal(seed=seed)

    if network["activation"] == "linear":
        activation = linear()
        kerner_initializer = RandomUniform(seed=seed)
    elif network["activation"] == "elu":
        activation = elu()
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "selu":
        activation = selu()
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "relu":
        activation = ReLU()
        kerner_initializer = he_uniform(seed=seed)
    elif network["activation"] == "lrelu":
        activation = LeakyReLU(alpha=0.2)
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "tanh":
        activation = tanh()
        kerner_initializer = RandomUniform(seed=seed)
    elif network["activation"] == "sigmoid":
        activation = sigmoid()
        kerner_initializer = RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    #######
    if network["architecture"] == 1: #with 7 conv layers
        # This will input x & noise and will output Y.
        x0 = Input(shape=(network["x_input_size"]), dtype='float')
        x_output = conv_layers_noNorm(num_conv=15, x_inp=x0, act_fun=activation, filters_width=network["filters_width"],
                         lambda_loss_amount=network["reg_r"], pooling_size=2)
        x_output = Dense(128, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
              use_bias=True)(x_output)
        x_output = activation(x_output)
        x_output = Dropout(network["dropout"])(x_output)

        noise = Input(shape=(network["z_input_size"],))
        noise_output = Dense(128, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
              use_bias=True)(noise)
        if network["activation"] == "lrelu":
            noise_output = LeakyReLU(alpha=0.2) (noise_output)
        else:
            noise_output = ReLU()(noise_output)
            print('ReLU activation is used after noise dense layer')
        #noise_output = Dropout(network["dropout"])(noise_output)

        concat = concatenate([x_output, noise_output])

        output = Dense(128, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
              use_bias=True)(concat)
        output = activation(output)
        output = Dropout(network["dropout"])(output)
        '''for fc_i in range(4):
            output = Dense(64, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
                           use_bias=True)(output)
            output = activation(output)
            output = Dropout(network["dropout"])(output)'''

        output = Dense(32, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
                       use_bias=True)(output)
        output = activation(output)
        output = Dropout(network["dropout"])(output)
        output = Dense(1, activation="linear", kernel_initializer=random_normal)(output)
        model = Model(inputs=[noise, x0], outputs=output)
        print(model.summary())

    else:
        raise NotImplementedError("Architecture does not exist")
    return model

def build_discriminator(network):
    seed = network["seed"]
    random_uniform = RandomUniform(seed=seed)

    if network["activation"] == "linear":
        activation = linear()
        kerner_initializer = RandomUniform(seed=seed)
    elif network["activation"] == "elu":
        activation = elu()
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "selu":
        activation = selu()
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "relu":
        activation = ReLU()
        kerner_initializer = he_uniform(seed=seed)
    elif network["activation"] == "lrelu":
        activation = LeakyReLU(alpha=0.2)
        kerner_initializer = he_normal(seed=seed)
    elif network["activation"] == "tanh":
        activation = tanh()
        kerner_initializer = RandomUniform(seed=seed)
    elif network["activation"] == "sigmoid":
        activation = sigmoid()
        kerner_initializer = RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")
    ######
    if network["architecture"] == 1: #with 7 conv
        # This will input x & label and will output Y true or fake.
        x0 = Input(shape=(network["x_input_size"]), dtype='float')
        x_output = conv_layers_noNorm(num_conv=15, x_inp=x0, act_fun=activation, filters_width=network["filters_width"],
                         lambda_loss_amount=network["reg_r"], pooling_size=2)
        x_output = Dense(128, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
                       use_bias=True)(x_output)
        x_output = activation(x_output)
        x_output = Dropout(network["dropout"])(x_output)

        label = Input(shape=(network["y_input_size"],))
        label_output = Dense(128, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
              use_bias=True)(label)
        if network["activation"] == "lrelu":
            label_output = LeakyReLU(alpha=0.2)(label_output)
        else:
            label_output = ReLU()(label_output)
            print('ReLU activation is used after noise dense layer')
        label_output = Dropout(network["dropout"])(label_output)

        concat = concatenate([x_output, label_output])

        '''for fc_i in range(4):
            concat = Dense(64, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
                           use_bias=True)(concat)
            concat = activation(concat)
            concat = Dropout(network["dropout"])(concat)'''

        concat = Dense(32, kernel_initializer=kerner_initializer, kernel_regularizer=l2(network["reg_r"]),
                       use_bias=True)(concat)
        concat = activation(concat)
        concat = Dropout(network["dropout"])(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x0, label], outputs=validity)
    else:
        raise NotImplementedError("Architecture does not exist")
    print(model.summary())
    return model

def build_gan(generator, discriminator, latent_dim, window_length, numChannels, optimizer=Adam(lr=0.0001, beta_1=0.5)):
    '''
    Defines and compiles GAN model. It bassically chains Generator
    and Discriminator in an assembly-line sort of way where the input is
    the Generator's input. The Generator's output is the input of the Discriminator,
    which outputs the output of the whole GAN.

    Params:
        optimizer=Adam(0.0002, 0.5) - recommended values
    '''

    noise = Input(shape=(latent_dim,))
    features = Input(shape=(window_length, numChannels))

    labels_gen = generator([noise, features])

    # We freeze the discriminator's layers since we're only
    # interested in the generator and its learning
    discriminator.trainable = False
    valid = discriminator([features, labels_gen])

    model = Model([noise, features], valid)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model
