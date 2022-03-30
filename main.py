from dataset_v0_0 import create_chirps
from train_v0_0 import *
from networks_v0_0 import *
import os, pickle, random, math
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def calculate_results(Results_train, Results_val, GT_scores, pred_scores, save_dir):
    # plotting the training curves
    plt.figure()
    plt.plot(range(Results_train.shape[0]), Results_train[:, 3], 'b',
             range(Results_train.shape[0]),
             Results_val[:, 3], 'g--', linewidth=2)
    plt.ylabel('Correlation', fontsize='large', weight='bold')
    plt.xlabel('Trail #', fontsize='large', weight='bold')
    plt.legend(['Training', 'Validation'], fontsize='large')
    plt.grid(True)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.savefig(save_dir + '_corr_curve.png')


    plt.figure()
    plt.plot(range(Results_train.shape[0]), Results_train[:, 0], 'b',
             range(Results_train.shape[0]),
             Results_val[:, 0], 'g--', linewidth=2)
    plt.ylabel('RMSE', fontsize='large', weight='bold')
    plt.xlabel('Trail #', fontsize='large', weight='bold')
    plt.legend(['Training', 'Validation'], fontsize='large')
    plt.grid(True)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.savefig(save_dir + '_rmse_curve.png')

    #calculating metrics on testind data:
    Results_test = np.zeros((5))
    Results_test[0] = np.sqrt(mean_squared_error(GT_scores, pred_scores))
    Results_test[1] = mean_absolute_error(GT_scores, pred_scores)
    Results_test[2] = r2_score(GT_scores, pred_scores)
    corrI = stats.pearsonr(GT_scores, pred_scores)
    Results_test[3] = corrI[0]
    Results_test[4] = corrI[1]

    print(
        "Final testing results: RMSE %.2f, MAE %.2f, R2 score %.2f, Correlation coefficient %.2f (p=%.4f)." % (
            Results_test[0],
            Results_test[1],
            Results_test[2],
            Results_test[3], Results_test[4]))


    #plotting gold-standard vs estimated scores
    plt.figure()
    xfit = np.array([np.min(GT_scores), np.max(GT_scores)])
    plt.plot(GT_scores, pred_scores, 'ko', xfit, xfit,
             'b--')
    plt.ylabel('Estimated scores', fontsize='large', weight='bold')
    plt.xlabel('Gold-standard scores', fontsize='large', weight='bold')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.grid(True)
    plt.savefig(save_dir + '_gold_vs_estimated.png')
    plt.close()

    return Results_test



if __name__=='__main__':
######################################## General parameters################################################
    data_file = 'chirp_data_v0_0.pkl'
    numChannels = 1
    SaveDirParent = './main_results/'
######################################## Loading and preparing data#################################################
    print('Loading pickled synthetic data files if available')
    if os.path.isfile('./' + data_file) == True:
        with open('./' + data_file, 'rb') as data:
            [data_chirp_train, win_info_train, data_chirp_test, win_info_test, fs, window_length,
             freq_range] = pickle.load(data)
        nRounds_train = win_info_train.shape[0]
        nRounds_test = win_info_test.shape[0]
    else: #else create the dataset
        data_chirp_train, win_info_train, data_chirp_test, win_info_test, fs, window_length, freq_range = create_chirps(
            freq_range=[0, 20], freq_removed=[5, 8], num_wins=1000, fs=50, window_sec=5, data_file=data_file
        )
        nRounds_train = win_info_train.shape[0]
        nRounds_test = win_info_test.shape[0]
####################################### Splitting training data into training and validation, and normalization#####
    # Creating training and validation splits (80%,20%) from training folds #
    val_perc = 0.2
    random.seed(100)
    trainInds = random.sample(range(0, nRounds_train), nRounds_train)
    random.seed()
    data_chirp_val = data_chirp_train[trainInds[0:math.ceil(nRounds_train * val_perc)]]
    win_info_val = win_info_train[trainInds[0:math.ceil(nRounds_train * val_perc)]]
    nRounds_val = data_chirp_val.shape[0]  # number of rounds for training
    data_chirp_train = data_chirp_train[
        trainInds[math.floor(nRounds_train * val_perc) + 1:nRounds_train]]
    win_info_train = win_info_train[trainInds[math.floor(nRounds_train * val_perc) + 1:nRounds_train]]
    nRounds_train = data_chirp_train.shape[0]  # number of windows for training

    GT_train = win_info_train[:, 1]  # Chirp end-frequecy
    GT_val = win_info_val[:, 1]  # Chirp end-frequecy
    GT_test = win_info_test[:, 1]# Chirp end-frequecy

    #normalize training data
    meanBothChann = np.nanmean(data_chirp_train)
    stdBothChann = np.nanstd(data_chirp_train)
    data_chirp_train = (data_chirp_train - meanBothChann) / stdBothChann
    #normalize validation data
    data_chirp_val = (data_chirp_val - meanBothChann) / stdBothChann
    #normalize testing data
    data_chirp_test = (data_chirp_test - meanBothChann) / stdBothChann



########################### training the CNN model ################################################################
    ############### CNN training parameters####
    num_conv_layers = 7
    num_fc_layers = 2
    filters_width = 4
    pooling_size = 2
    batch_size = 64
    noEpochs = 200
    savingTime = 1  # save after N epochs
    base_learning_rate = 0.0005
    decayRate = base_learning_rate / noEpochs
    momentumValue = 0.9
    dropout_Prob = 0.5  # was 0.5
    lambda_loss_amount = 0.0  # 0.0005 or weight decay. As a rule of thumb, the more training examples you have, the
    # weaker this term should be. The more parameters you have (i.e., deeper net, larger filters, larger InnerProduct
    # layers etc.) the higher this term should be.
    save_path_loss = SaveDirParent + "cnn_conv" + str(num_conv_layers) + '_fc' + str(num_fc_layers) + ".h5"

    print('Training the CNN model:')
    cnn_model = build_cnn_fun(num_conv_layers, window_length, numChannels, filters_width, pooling_size,
              dropout_Prob, lambda_loss_amount)
    print(cnn_model.summary())
    cnn_model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=base_learning_rate, decay=decayRate), metrics=['mae'])
    loss_train, loss_val, Results_train, Results_val = train_cnn(
        cnn_model, data_chirp_train, GT_train, data_chirp_val, GT_val, batch_size, noEpochs, savingTime,
        save_path_loss)

    ############### printing and plotting the CNN results###
    cnn_model = load_model(save_path_loss)
    pred_scores = cnn_model.predict(data_chirp_test)[:, 0]
    print('Printing and plotting the CNN results:')
    Results_test = calculate_results(Results_train, Results_val, GT_test, pred_scores, save_dir=SaveDirParent + 'cnn')


    ############### Saving the variables####################
    with open(SaveDirParent + "cnn_conv" + str(num_conv_layers) + '_fc' + str(num_fc_layers) + '_results.pkl',
              'wb') as Results:  # Python 3: open(..., 'wb')
        pickle.dump(
            [Results_train, Results_val, loss_train, loss_val, Results_test, pred_scores, GT_test], Results)



    ########################### training the cGAN model ################################################################
    ############### cGAN training parameters####
    arch_i = 1
    learning_rate_g = 0.0001  #
    learning_rate_d = 0.0001
    latent_dim = 100 # dimension of the latent space
    n_iterations = 35000 #each iteration for a batch not all training data
    log_hist_save = 100 #save and print metrics every n iterations
    num_rep_test = 10 #repeating prediction of test data n times then averaging

    network_d = {
        "architecture": arch_i,
        "window_length":window_length,
        "numChannels": numChannels,
        "filters_width": filters_width,
        "pooling_size": pooling_size,
        "seed": 1985,
        "activation": "lrelu",
        "dropout": 0.0,
        "reg_r": 0.0,
        "x_input_size": (window_length, numChannels),
        "y_input_size": 1,
        "z_input_size": latent_dim
    }
    network_g = {
        "architecture": arch_i,
        "window_length":window_length,
        "numChannels": numChannels,
        "filters_width": filters_width,
        "pooling_size": pooling_size,
        "seed": 1985,
        "activation": "relu",
        "dropout": 0.0,
        "reg_r": 0.0,
        "x_input_size": (window_length, numChannels),
        "y_input_size": 1,
        "z_input_size": latent_dim
    }
    save_path_loss = SaveDirParent + "cgan_arch" + str(arch_i) + ".h5"

    print('Training the cGAN model:')
    discriminator = build_discriminator(network_d)
    discriminator.compile(
        loss=['binary_crossentropy'],
        optimizer=Adam(lr=learning_rate_d, beta_1=0.5),
        metrics=['accuracy'])
    generator = build_generator(network_g)
    gan = build_gan(generator, discriminator, latent_dim, window_length, numChannels,
                    optimizer=Adam(lr=learning_rate_g, beta_1=0.5))
    loss_real_hist, acc_real_hist, \
    loss_fake_hist, acc_fake_hist, \
    loss_gan_hist, acc_gan_hist, \
    Results_train, Results_val = train_cgan(gan, generator, discriminator, X=data_chirp_train, y=GT_train,
                                           X_val=data_chirp_val, y_val=GT_val,
                                           n_iterations=n_iterations,
                                           batch_size=batch_size, hist_every=log_hist_save, log_every=log_hist_save,
                                           save_dir=save_path_loss, latent_dim=latent_dim)
    ############### printing and plotting the cGAN results###
    # Find the testing results for the model with highest validation Correlation.
    gan = load_model(save_path_loss)
    generator = gan.layers[2]
    pred_temp = np.zeros((nRounds_test,))
    for r in range(num_rep_test):
        noise = np.random.uniform(0, 1, (nRounds_test, latent_dim))
        pred_temp = pred_temp + generator.predict([noise, data_chirp_test])[:, 0]
    pred_scores = pred_temp / num_rep_test
    print('Printing and plotting the cGAN results:')
    Results_test = calculate_results(Results_train, Results_val, GT_test, pred_scores, save_dir=SaveDirParent + 'cgan')

    #GAN accuracy and Loss
    x_itr = np.arange(0, n_iterations / 1000, log_hist_save / 1000)

    # GAN accuracy and Loss
    ax, fig = plt.subplots(figsize=(15, 6))
    plt.plot(x_itr, loss_real_hist, linewidth=2.5)
    plt.plot(x_itr, loss_fake_hist, linewidth=2.5)
    plt.plot(x_itr, loss_gan_hist, linewidth=2.5)
    plt.title('Training loss over training iteration', fontsize=16)
    plt.ylabel('Binary crossentropy loss', fontsize=16)
    plt.xlabel('Iteration (k)', fontsize=16)
    plt.legend(['D-real', 'D-fake', 'G'])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(SaveDirParent + 'cgan_loss.png')

    ax, fig = plt.subplots(figsize=(15, 6))
    plt.plot(x_itr, acc_real_hist, linewidth=2.5)
    plt.plot(x_itr, acc_fake_hist, linewidth=2.5)
    plt.plot(x_itr, acc_gan_hist, linewidth=2.5)
    plt.title('Training accuracy over training iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Iteration (k)', fontsize=16)
    plt.legend(['D-real', 'D-fake', 'G'])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(SaveDirParent + 'cgan_acc.png')
    plt.close()

    ############### Saving the variables####################
    with open(SaveDirParent + "cgan_arch" + str(arch_i) + '_results.pkl',
              'wb') as Results:  # Python 3: open(..., 'wb')
        pickle.dump(
            [Results_train, Results_val, loss_real_hist, acc_real_hist, loss_fake_hist, acc_fake_hist,
             loss_gan_hist, acc_gan_hist, Results_test, pred_scores, GT_test], Results)
