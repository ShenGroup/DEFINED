import numpy as np
import torch
from torch.utils.data import Dataset


def generate_rician_channel(num_ant, K_factor=1.0):
    """Generate a single Rician MIMO channel matrix H of shape (num_ant, num_ant)."""
    H_scatter = (
        np.random.randn(num_ant, num_ant)
        + 1j * np.random.randn(num_ant, num_ant)
    ) / np.sqrt(2)

    H_los = (
        np.ones((num_ant, num_ant))
        + 1j * np.ones((num_ant, num_ant))
    ) / np.sqrt(2)

    H = (
        np.sqrt(K_factor / (K_factor + 1.0)) * H_los
        + np.sqrt(1.0 / (K_factor + 1.0)) * H_scatter
    )
    return H


def generate_signals(batch_size, args, channel_type="rayleigh", K_factor=1.0):
    rng = np.random.default_rng()

    X, Y, Hs = [], [], []

    for _ in range(batch_size):
        if channel_type.lower() == "rayleigh":
            H = (
                rng.standard_normal((args.num_ant, args.num_ant))
                + 1j * rng.standard_normal((args.num_ant, args.num_ant))
            ) / np.sqrt(2)
        elif channel_type.lower() == "rician":
            H = generate_rician_channel(args.num_ant, K_factor=K_factor)
        else:
            raise ValueError("channel_type must be either 'rayleigh' or 'rician'")
        Hs.append(H)

        snr_db = rng.uniform(args.SNR_dB_min, args.SNR_dB_max)
        noise_var = 10.0 ** (-snr_db / 10.0)

        x, _ = generate_modulated_signal(args, args.modulation, rng)

        n = (
            np.sqrt(noise_var)
            * (
                rng.standard_normal((args.num_ant, args.prompt_seq_length))
                + 1j
                * rng.standard_normal((args.num_ant, args.prompt_seq_length))
            )
            / np.sqrt(2)
        )
        y = H @ x + n  # (num_ant, T)

        X.append(x.T)  # (T, num_ant)
        Y.append(y.T)

    return np.stack(X), np.stack(Y), Hs


def generate_modulated_signal(args, modulation, rng=None):
    """Generate a baseband modulated signal sequence (complex)."""
    if rng is None:
        rng = np.random.default_rng()

    if modulation in { "16QAM", "64QAM"}:
        # Same logic as original code: '4QAM' -> 4, '16QAM' -> 16, ...
        M = int(modulation[:-3])
        side = int(np.sqrt(M))

        const_1d = 2 * np.arange(side) - (side - 1)
        constellation = const_1d + 1j * const_1d[:, np.newaxis]
        constellation = constellation.flatten()

        # Normalize average power to 1
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))

        symbols = rng.choice(
            constellation,
            size=(args.num_ant, args.prompt_seq_length),
        )

    elif modulation == "BPSK":
        constellation = np.array([1, -1])
        symbols = rng.choice(
            [1, -1],
            size=(args.num_ant, args.prompt_seq_length),
        )
    elif modulation == "QPSK":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        symbols = rng.choice(
            constellation,
            size=(args.num_ant, args.prompt_seq_length),
        )
    else:
        raise ValueError("Unsupported modulation type: {}".format(modulation))

    return symbols, constellation


def count_modulation_symbols(args):
    """Return the constellation size for the given modulation.

    Follows the logic of the original implementation, including
    the special case for MIMO == 2.
    """
    modulation = args.modulation
    MIMO = args.num_ant

    if MIMO == 2:
        if modulation == "BPSK":
            return 4
        elif modulation == "QPSK":
            return 16
        elif modulation == "16QAM":
            return 256
        else:
            raise ValueError("Unsupported modulation type")

    if modulation in {"4QAM", "16QAM", "64QAM"}:
        M = int(modulation[:-3])
        return M
    elif modulation == "BPSK":
        return 2
    elif modulation == "QPSK":
        return 4
    else:
        raise ValueError("Unsupported modulation type: {}".format(modulation))



def complex_to_vec(X):
    """Convert complex array X to a real-valued array [Re(X), Im(X)]."""
    return np.concatenate([np.real(X), np.imag(X)], axis=-1)



def lmmse_channel_estimation(x, y, snr):
    """Perform LMMSE-like channel estimation (matching the original code).

    Note
    ----
    The covariance term follows the original implementation:
        cov = X^H X + sigma2
    i.e. a scalar offset rather than sigma2 * I.
    """
    sigma2 = 10.0 ** (-snr / 10.0)
    cov = np.conj(x).T @ x + sigma2
    inv_term = np.linalg.pinv(cov)
    h_est = inv_term @ np.conj(x).T @ y
    return h_est.T


def predict_symbol(h_est, y, constellation):
    """Predict transmit symbol via exhaustive search (ML detection).

    h_est : (num_ant, num_ant)
    y     : (num_ant,)
    constellation : (M,) complex
    """
    num_ant = h_est.shape[0]

    # joint constellation over all antennas
    all_combinations = np.array(
        np.meshgrid(*[constellation] * num_ant)
    ).T.reshape(-1, num_ant)

    y_col = y.reshape(-1, 1)
    predicted_y = h_est @ all_combinations.T
    distances = np.sum(np.abs(y_col - predicted_y) ** 2, axis=0)
    best_idx = np.argmin(distances)
    return all_combinations[best_idx]


def build_joint_constellation(modulation, num_ant):
    """Build the joint constellation over all antennas.

    The order is deterministic and depends only on modulation and num_ant,
    so encoding is stable across different runs.
    """
    # single-antenna constellation
    if modulation in {"16QAM", "64QAM"}:
        M = int(modulation[:-3])
        side = int(np.sqrt(M))
        const_1d = 2 * np.arange(side) - (side - 1)
        constellation = const_1d + 1j * const_1d[:, np.newaxis]
        constellation = constellation.flatten()
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    elif modulation == "BPSK":
        constellation = np.array([1, -1])
    elif modulation == "QPSK":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    else:
        raise ValueError("Unsupported modulation type: {}".format(modulation))

    # joint constellation: Cartesian product over antennas
    joint = np.array(
        np.meshgrid(*[constellation] * num_ant)
    ).T.reshape(-1, num_ant)

    return joint  # shape (M_joint, num_ant)


def encode_joint_symbols(x_complex, joint_constellation):
    """Map complex MIMO symbols to joint constellation indices.

    x_complex           : (B, T, num_ant) or (T, num_ant), complex
    joint_constellation : (M_joint, num_ant), complex

    Returns
    -------
    indices : np.ndarray
        Shape (B, T) or (T,), each entry is in [0, M_joint).
    """
    if x_complex.ndim == 2:
        # (T, num_ant) -> (1, T, num_ant)
        x_complex = x_complex[None, :, :]

    B, T, num_ant = x_complex.shape
    M_joint = joint_constellation.shape[0]

    # flatten to (B*T, num_ant)
    flat_x = x_complex.reshape(-1, num_ant)

    # compute distances to all joint constellation points
    diff = flat_x[:, None, :] - joint_constellation[None, :, :]
    dist2 = np.abs(diff) ** 2
    dist2_sum = dist2.sum(axis=-1)          # (B*T, M_joint)

    indices = np.argmin(dist2_sum, axis=1)  # (B*T,)
    indices = indices.reshape(B, T)

    if indices.shape[0] == 1:
        return indices[0]   # (T,)
    return indices          # (B, T)


def one_hot_from_indices(indices, num_classes):
    """Convert integer indices to one-hot coding.

    indices:
        (B, T) or (T,)
    returns:
        (B, T, num_classes) or (T, num_classes)
    """
    eye = np.eye(num_classes, dtype=np.float32)

    if indices.ndim == 1:
        return eye[indices]             # (T, num_classes)
    return eye[indices]                 # (B, T, num_classes)



class MIMOSequenceDataset(Dataset):
    """Dataset that generates sequences and returns:

        x : (T, num_classes), float32 one-hot (joint constellation, transmit symbols)
        y : (T, 2 * num_ant), float32 real-valued features [Re(y), Im(y)]

    Each sample also includes:
        H              (num_ant, num_ant), complex64 tensor
        snr_db         scalar, float32 tensor
        joint_const    (num_classes, num_ant), complex64 tensor
    """

    def __init__(self, args, num_samples,
                 joint_constellation,
                 channel_type="rayleigh",
                 K_factor=1.0,
                 seed=None):
        """
        Parameters
        ----------
        args : argparse.Namespace
            Should provide num_ant, prompt_seq_length, SNR_dB_min, SNR_dB_max, modulation.
        num_samples : int
            Length of the dataset (__len__).
        joint_constellation : np.ndarray
            Precomputed joint constellation, shape (num_classes, num_ant), complex.
            Use build_joint_constellation(modulation, num_ant) to obtain this.
        channel_type : str
            'rayleigh' or 'rician'.
        K_factor : float
            Rician K-factor when using 'rician'.
        seed : int or None
            Random seed for this dataset.
        """
        super().__init__()
        self.args = args
        self.num_samples = num_samples
        self.channel_type = channel_type.lower()
        self.K_factor = K_factor

        self.rng = np.random.default_rng(seed)

        if self.channel_type not in {"rayleigh", "rician"}:
            raise ValueError("channel_type must be either 'rayleigh' or 'rician'")

        # Store joint constellation in both numpy and tensor form
        self.joint_constellation_np = np.asarray(joint_constellation)
        self.num_classes = self.joint_constellation_np.shape[0]
        self.joint_constellation_tensor = torch.from_numpy(
            self.joint_constellation_np.astype(np.complex64)
        )

        print(
            "*** MIMOSequenceDataset initialized with {} channels, num_classes={} ***".format(
                self.channel_type, self.num_classes
            )
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.channel_type == "rayleigh":
            H = (
                self.rng.standard_normal((self.args.num_ant, self.args.num_ant))
                + 1j
                * self.rng.standard_normal((self.args.num_ant, self.args.num_ant))
            ) / np.sqrt(2)
        else:  # 'rician'
            H = generate_rician_channel(self.args.num_ant, K_factor=self.K_factor)

        snr_db = self.rng.uniform(self.args.SNR_dB_min, self.args.SNR_dB_max)
        noise_var = 10.0 ** (-snr_db / 10.0)

        x, _ = generate_modulated_signal(
            self.args, self.args.modulation, self.rng
        )  # x: (num_ant, T)

        n = (
            np.sqrt(noise_var)
            * (
                self.rng.standard_normal((self.args.num_ant, self.args.prompt_seq_length))
                + 1j
                * self.rng.standard_normal(
                    (self.args.num_ant, self.args.prompt_seq_length)
                )
            )
            / np.sqrt(2)
        )
        y = H @ x + n  # (num_ant, T)

        x_seq = x.T      # (T, num_ant), complex
        y_seq = y.T      # (T, num_ant), complex

        x_indices = encode_joint_symbols(x_seq, self.joint_constellation_np)  # (T,)

        x_onehot_np = one_hot_from_indices(x_indices, self.num_classes)       # (T, C)

        y_feat_np = complex_to_vec(y_seq)                                     # (T, 2N)

        x_tensor = torch.from_numpy(x_onehot_np.astype(np.float32))     # (T, C)
        y_tensor = torch.from_numpy(y_feat_np.astype(np.float32))       # (T, 2N)
        H_tensor = torch.from_numpy(H.astype(np.complex64))             # (N, N)
        snr_tensor = torch.tensor(snr_db, dtype=torch.float32)

        return {
            "x": x_tensor,                       # (T, num_classes), float32
            "y": y_tensor,                       # (T, 2 * num_ant), float32
            "H": H_tensor,                       # (num_ant, num_ant), complex64
            "snr_db": snr_tensor,                # ()
            "joint_constellation": self.joint_constellation_tensor,  # (C, num_ant)
        }