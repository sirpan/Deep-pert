import os
import re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional, Dict

# Keras compatibility imports (keras or tensorflow.keras)
try:
    from keras.layers import Input, Dense, Lambda, Activation, Dropout, BatchNormalization, GaussianNoise  # type: ignore
    from keras.models import Model  # type: ignore
    from keras import backend as K  # type: ignore
    from keras import metrics, optimizers  # type: ignore
    from keras.callbacks import Callback  # type: ignore
except Exception:
    from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Dropout, BatchNormalization, GaussianNoise  # type: ignore
    from tensorflow.keras.models import Model  # type: ignore
    from tensorflow.keras import backend as K  # type: ignore
    from tensorflow.keras import metrics, optimizers  # type: ignore
    from tensorflow.keras.callbacks import Callback  # type: ignore

from sklearn.metrics import mean_squared_error, r2_score


# TensorFlow session configuration (default: allow growth)
try:
    _tf_config = tf.ConfigProto()
    _tf_config.gpu_options.allow_growth = True
    _sess = tf.Session(config=_tf_config)
    # Not all backends expose set_session
    if hasattr(K, 'set_session'):
        K.set_session(_sess)
except Exception:
    pass


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs=None):
        current_beta = K.get_value(self.beta)
        if current_beta < 1:
            K.set_value(self.beta, min(current_beta + self.kappa, 1))


class CosineAnnealingLR(Callback):
    def __init__(self, lr_max, lr_min, cycle_length, verbose=0):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cycle_length = cycle_length
        self.verbose = verbose
        self.history = {"lr": []}

    def on_epoch_begin(self, epoch, logs=None):
        t_cur = epoch % self.cycle_length
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t_cur / self.cycle_length))
        K.set_value(self.model.optimizer.lr, lr)
        self.history["lr"].append(lr)
        if self.verbose > 0:
            print(f"\nEpoch {epoch}: Learning rate {lr:.6f}")


def _auto_find_input_path(input_base: str, cancer_type: str, pca_method: str) -> Tuple[str, int]:
    pattern = os.path.join(input_base, f"{cancer_type}_DATA_TOP2_JOINED_{pca_method}_*L.tsv")
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")
    if len(matching_files) > 1:
        raise ValueError(f"找到多个匹配文件: {matching_files}")
    input_path = matching_files[0]
    filename = os.path.basename(input_path)
    dim_match = re.search(fr"{pca_method}_(\d+)L", filename)
    if not dim_match:
        raise ValueError("文件名格式错误，无法提取维度值")
    original_dim = int(dim_match.group(1))
    return input_path, original_dim


def _build_ae(original_dim: int, dim1: int, latent_dim: int) -> Model:
    init_mode = 'glorot_uniform'
    dropout = 0.1
    x = Input(shape=(original_dim,))
    net = Dense(dim1, kernel_initializer=init_mode)(x)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(dropout)(net)
    core = Dense(latent_dim, kernel_initializer=init_mode, name='encoder_output')(net)
    decoder_h = Dense(dim1, activation='relu', kernel_initializer=init_mode)
    d2 = Dropout(dropout)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    h_decoded = decoder_h(core)
    h_decoded2 = d2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded2)
    return Model(x, x_decoded_mean)


def _build_dae(original_dim: int, dim1: int, latent_dim: int, noise_std: float = 1.0) -> Model:
    # Denoising AE: inject Gaussian noise on inputs during training
    init_mode = 'glorot_uniform'
    dropout = 0.1
    x = Input(shape=(original_dim,))
    noisy = GaussianNoise(noise_std)(x)
    net = Dense(dim1, kernel_initializer=init_mode)(noisy)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(dropout)(net)
    core = Dense(latent_dim, kernel_initializer=init_mode, name='encoder_output')(net)
    decoder_h = Dense(dim1, activation='relu', kernel_initializer=init_mode)
    d2 = Dropout(dropout)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    h_decoded = decoder_h(core)
    h_decoded2 = d2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded2)
    return Model(x, x_decoded_mean)


def _build_vae(original_dim: int, dim1: int, dim2: int, latent_dim: int) -> Tuple[Model, K.variable, K.variable]:
    init_mode = 'glorot_uniform'
    x = Input(shape=(original_dim,))
    net = Dense(dim1, kernel_initializer=init_mode)(x)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dense(dim2, kernel_initializer=init_mode)(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    z_mean = Dense(latent_dim, kernel_initializer=init_mode)(net)
    z_log_var = Dense(latent_dim, kernel_initializer=init_mode)(net)
    z = Lambda(sampling, name='sampling_layer')([z_mean, z_log_var])

    decoder_h = Dense(dim1, activation='relu', kernel_initializer=init_mode)
    decoder_h2 = Dense(dim2, activation='relu', kernel_initializer=init_mode)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    h_decoded = decoder_h(z)
    h_decoded2 = decoder_h2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded2)

    vae = Model(x, x_decoded_mean)

    beta = K.variable(1.0)
    kappa = 0.01

    def vae_loss(x_true, x_dec):
        reconstruction_loss = original_dim * metrics.mse(x_true, x_dec)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    vae._vae_custom_objects = {"vae_loss": vae_loss}
    return vae, beta, kappa


def _compile_and_train(model: Model,
                       loss_fn,
                       df_values: np.ndarray,
                       epochs: int,
                       batch_size: int,
                       lr_max: float,
                       lr_min: float,
                       cycle_length: int,
                       extra_callbacks: Optional[list] = None):
    optimizer = optimizers.Adam(lr=lr_max)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metrics.mse])
    callbacks = [CosineAnnealingLR(lr_max, lr_min, cycle_length, verbose=1)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    history = model.fit(
        df_values, df_values,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    return history


def _save_components(model: Model,
                     latent_dim: int,
                     df: pd.DataFrame,
                     output_folder: str,
                     run: int,
                     encoder_output_name: str) -> Tuple[float, float]:
    encoder = Model(model.input, model.get_layer(encoder_output_name).output)
    latent_input = Input(shape=(latent_dim,))
    decoder_layers = [model.layers[-3], model.layers[-2], model.layers[-1]]
    x = decoder_layers[0](latent_input)
    x = decoder_layers[1](x)
    decoded_output = decoder_layers[2](x)
    decoder = Model(latent_input, decoded_output)

    os.makedirs(output_folder, exist_ok=True)

    model_json = encoder.to_json()
    with open(os.path.join(output_folder, f'encoder_{latent_dim}L_run{run}.json'), 'w') as jf:
        jf.write(model_json)
    encoder.save_weights(os.path.join(output_folder, f'encoder_{latent_dim}L_run{run}.h5'))

    model_json = decoder.to_json()
    with open(os.path.join(output_folder, f'dncoder_{latent_dim}L_run{run}.json'), 'w') as jf:
        jf.write(model_json)
    decoder.save_weights(os.path.join(output_folder, f'dncoder_{latent_dim}L_run{run}.h5'))

    latent_df = pd.DataFrame(encoder.predict(df.values), index=df.index)
    latent_df.to_csv(os.path.join(output_folder, f'latent_{latent_dim}L_run{run}.tsv'), sep='\t')

    reconstructed = decoder.predict(latent_df.values)
    mse = mean_squared_error(df.values, reconstructed)
    r2 = r2_score(df.values.flatten(), reconstructed.flatten())
    print(f"\nEvaluation Run {run}:\nMSE: {mse:.4f}\nR²: {r2:.4f}")
    return mse, r2


def run_model(model_type: str,
              cancer_type: str,
              dim1: int,
              dim2: int,
              latent_dim: int,
              run: int,
              input_base: str,
              output_base: str,
              epochs: int = 100,
              batch_size: int = 256,
              lr_max: float = 1e-3,
              lr_min: float = 1e-6,
              cycle_length: int = 20,
              result_collector: Optional[Dict] = None,
              gpu_number: Optional[str] = None,
              pca_method: str = 'PCA',
              noise_std: float = 1.0) -> Tuple[float, float]:
    """Unified trainer for AE, DAE, VAE.

    model_type: 'AE' | 'DAE' | 'VAE'
    """
    if gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Seed
    np.random.seed(123456 * run)
    try:
        tf.set_random_seed(123456 * run)  # TF1.x
    except AttributeError:
        tf.random.set_seed(123456 * run)  # TF2.x

    # Load data
    input_path, original_dim = _auto_find_input_path(input_base, cancer_type, pca_method)
    print(f"\n自动识别输入维度: {original_dim}")
    df = pd.read_csv(input_path, sep='\t', index_col=0)
    print(f"数据已加载 | 维度: {df.shape} | 示例细胞: {df.index.tolist()[:3]}...")

    # Build and train
    if model_type.upper() == 'AE':
        model = _build_ae(original_dim, dim1, latent_dim)
        def loss_fn(x_true, x_dec):
            return K.mean(original_dim * metrics.mse(x_true, x_dec))
        history = _compile_and_train(model, loss_fn, df.values, epochs, batch_size, lr_max, lr_min, cycle_length)
        encoder_name = 'encoder_output'
    elif model_type.upper() == 'DAE':
        model = _build_dae(original_dim, dim1, latent_dim, noise_std=noise_std)
        def loss_fn(x_true, x_dec):
            return K.mean(original_dim * metrics.mse(x_true, x_dec))
        history = _compile_and_train(model, loss_fn, df.values, epochs, batch_size, lr_max, lr_min, cycle_length)
        encoder_name = 'encoder_output'
    elif model_type.upper() == 'VAE':
        vae, beta, kappa = _build_vae(original_dim, dim1, dim2, latent_dim)
        def loss_fn(x_true, x_dec):
            # Placeholder; actual loss attached within _build_vae via closure
            return vae._vae_custom_objects["vae_loss"](x_true, x_dec)
        history = _compile_and_train(
            vae,
            loss_fn,
            df.values,
            epochs,
            batch_size,
            lr_max,
            lr_min,
            cycle_length,
            extra_callbacks=[WarmUpCallback(beta, kappa)],
        )
        model = vae
        encoder_name = 'sampling_layer'
    else:
        raise ValueError("model_type must be one of: AE, DAE, VAE")

    output_folder = os.path.join(output_base, cancer_type)
    final_loss = history.history['loss'][-1]
    final_mse = history.history['mean_squared_error'][-1]
    eval_mse, eval_r2 = _save_components(model, latent_dim, df, output_folder, run, encoder_name)

    if result_collector is not None:
        if cancer_type not in result_collector:
            result_collector[cancer_type] = []
        result_collector[cancer_type].append({
            'run': run,
            'train_loss': final_loss,
            'train_mse': final_mse,
            'eval_mse': eval_mse,
            'eval_r2': eval_r2,
        })

    K.clear_session()
    return eval_mse, eval_r2


