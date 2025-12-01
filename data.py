import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.constants as const
from matplotlib.ticker import LogLocator, NullFormatter
from scipy import linalg
from scipy.linalg import toeplitz
from scipy.special import j0
from scipy.stats import mode

from parameters import parameter_reading




# modulation =['4QAM', '16QAM', '64QAM', '2PSK', '8PSK']
def generate_signals(batch_size, n_tasks, args, seed_task=0, seed_example=0):
    # np.random.seed(seed_task)
    modulation = args.modulation
    # # Rayleigh fading channel
    H = ( np.random.randn( n_tasks, args.num_ant, args.num_ant ) + 1j * np.random.randn (
        n_tasks, args.num_ant, args.num_ant)) / np.sqrt(2)
    print("***generate Rayleigh Channel***")


    SNR_in_dB_all = np.random.rand(n_tasks) * (args.SNR_dB_max - args.SNR_dB_min) + args.SNR_dB_min
    X, Y, Y_q, SNR = [], [], [], []
    np.random.seed(seed_example)
    Hs = []
    for ii in range(batch_size):
        id_rand = int(np.random.randint(0, n_tasks))
        H_eval = H[id_rand]
        Hs.append(H_eval)
        SNR_in_dB = SNR_in_dB_all[id_rand]
        noiseVar = 10 ** (-SNR_in_dB / 10)
        x,constellation = generate_modulated_signal(args, modulation)
        n = np.sqrt(noiseVar) * (np.random.randn(args.num_ant, args.prompt_seq_length) + 1j * np.random.randn( args.num_ant, args.prompt_seq_length)) / np.sqrt(2)
        y = H_eval @ x + n

        X.append(np.transpose(x))
        Y.append(np.transpose(y))
        Y_q.append(np.transpose(y_q))

        SNR.append(np.transpose(x * 0) + SNR_in_dB / 30 + 1j * (np.transpose(x * 0) + SNR_in_dB / 30))

    X = np.stack(X)
    Y = np.stack(Y)
    Y_q = np.stack(Y_q)
    return X, Y, Hs
    # return X, Y, Hs, SNR



def generate_modulated_signal(args, modulation):
    if modulation == '4QAM' or modulation == '16QAM' or modulation == '64QAM':
        # QAM Modulation
        M = int(modulation[:-3])  # Extract the number, e.g., 16 from '16QAM'
        constellation = (2 * np.arange(np.sqrt(M)) - (np.sqrt(M) - 1))
        constellation = constellation + 1j * constellation[:, np.newaxis]
        constellation = constellation.flatten()
        constellation = np.sqrt(np.mean(np.abs(constellation) ** 2))
        symbols = np.random.choice(constellation, size=(args.num_ant, args.prompt_seq_length))


    elif modulation == '2PSK':
        # BPSK Modulation
        # angles = np.array([0, np.pi])  # 2PSK corresponds to 0 and 180 degrees
        # constellation = np.exp(1j * angles)  # Convert angles to complex numbers on the unit circle
        constellation = np.array([1, -1])
        symbols = np.random.choice([1, -1], size=(args.num_ant, args.prompt_seq_length))

    elif modulation == '8PSK':
        # 8PSK Modulation
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        constellation = np.exp(1j * angles)
        # 8PSK符号自然位于单位圆上，平均功率已经是1
        symbols = np.random.choice(constellation, size=(args.num_ant, args.prompt_seq_length))
    elif modulation == '32APSK':
        # 32APSK Modulation
        # Define two circles with different radii and number of points
        inner_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        inner_radius = 0.5  # Example radius
        outer_radius = 1.0  # Example radius
        inner_constellation = inner_radius * np.exp(1j * inner_angles)
        outer_constellation = outer_radius * np.exp(1j * outer_angles)
        constellation = np.concatenate((inner_constellation, outer_constellation))
        # Normalize to ensure average power is 1
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
        symbols = np.random.choice(constellation, size=(args.num_ant, args.prompt_seq_length))
    else:
        raise ValueError("Unsupported modulation type")
    return symbols, constellation


def count_modulation_symbols(args):

    modulation = args.modulation
    MIMO = args.num_ant
    if MIMO == 2:
        if modulation == '2PSK':
            return 4
        elif modulation == '4QAM':
            return 16
        elif modulation == '16QAM':
            return 256
        else:
            raise ValueError("Unsupported modulation type")
    if modulation == '4QAM' or modulation == '16QAM' or modulation == '64QAM':
        # QAM modulation
        M = int(modulation[:-3])  # Extract the number, e.g., 16 from '16QAM'
        return M  # Number of symbols is the square of the side length of the grid
    elif modulation == '2PSK':
        # 2PSK has 2 symbols
        return 2
    elif modulation == '8PSK':
        # 8PSK has 8 symbols
        return 8
    elif modulation == '32APSK':
        # 32APSK has 32 symbols
        return 32
    else:
        raise ValueError("Unsupported modulation type")



# MMSE of h, use h to estimate x
def calculate_ser(args, nsample, length, SNR):
    # Update args
    args.prompt_seq_length = length+1
    args.SNR_dB_min = SNR
    args.SNR_dB_max = SNR
    ntask = 60000

    X, Y,_ ,Hs = generate_signals(nsample, ntask, args)
    _, constellation = generate_modulated_signal(args, args.modulation)

    errors = 0
    total_symbols = nsample

    for i in range(nsample):
        # Use the first 'length' symbols as pilots
        x_pilot = X[i, :length, :]
        y_pilot = Y[i, :length, :]
        h = Hs[i]


        h_est = lmmse_channel_estimation(x_pilot, y_pilot, SNR)
        squared_diff = np.sum(np.abs(h_est - h) ** 2)

        y_pred = Y[i, length, :]
        x_true = X[i, length, :]
        x_pred = predict_symbol(h_est, y_pred, constellation)

        if not np.allclose(x_pred, x_true, atol=1e-5):
            errors += 1

    ser = errors / total_symbols
    print(f"MIMO_{args.num_ant} {args.modulation} Pilot_{length}  SNR_{SNR} dB h_MMSE SER_"f"{ser:.4f}")
    return ser


def DFE_MMSE_SER(args, nsample, length, SNR):
    task =  f"DFE-MMSE-MIMO_{args.num_ant} {args.modulation} Pilot_{length}  SNR_{SNR} dB"
    print("***Start "+task)
    # Update args
    args.prompt_seq_length = 31
    args.SNR_dB_min = SNR
    args.SNR_dB_max = SNR
    ntask = 60000

    X, Y,_, Hs = generate_signals(nsample, ntask, args)
    assert X.shape == (nsample, args.prompt_seq_length, args.num_ant)
    assert Y.shape == (nsample, args.prompt_seq_length, args.num_ant)
    # Get constellation
    _, constellation = generate_modulated_signal(args, args.modulation)

    errors = np.zeros(args.prompt_seq_length)
    total_symbols = nsample

    for i in range(nsample):
        x_pilot = X[i, :length, :]
        y_pilot = Y[i, :length, :]
        for j in range(length , 31):
            h_est = lmmse_channel_estimation(x_pilot, y_pilot, SNR)
            y_pred = Y[i, j, :]
            x_true = X[i, j, :]
            x_pred = predict_symbol(h_est, y_pred, constellation)
            x_pilot = np.vstack((x_pilot, x_pred))
            y_pilot = np.vstack((y_pilot, y_pred))
            if not np.allclose(x_pred, x_true, atol=1e-5):
                errors[j] += 1

    ser = errors / total_symbols
    print("ser= ", np.round(ser, 4).tolist())

    time_steps = list(range(1, 31))
    DFE_MMSE = ser[1:]
    ser_str = np.array2string(ser, precision=3, separator=', ', suppress_small=True)
    name = result_string = f"MIMO_{args.num_ant} {args.modulation} Pilot_{length}  SNR_{SNR} dB"
    print(name)

    plt.figure(figsize=(10, 6)) 
    line_ICL, = plt.plot(time_steps, DFE_MMSE, color='C0', marker='.', label='DFE-MMSE')
    for x, y in zip(time_steps, DFE_MMSE):
        plt.annotate(f'{y:.3f}', (x, y), color='C0', textcoords="offset points", xytext=(-9, -23), ha='center',
                     fontsize=9, rotation=45)
    plt.title(task+ '\n \n' +"Symbol Error Rate", fontsize=11) 
    plt.xlabel('Pilot Sequence Length', fontsize=11)  
    plt.ylabel('Symbol Error Rate', fontsize=11)  
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))  
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))  
    ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}")) # type: ignore
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}")) # type: ignore
    plt.legend() 
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    ymin, ymax = plt.ylim()
    plt.ylim(ymin * 0.97, ymax)
    plt.tight_layout()  
    time_id = time.strftime("%m%d%H%M%S", time.localtime())
    plt.savefig(f'../Results_v2/DFE-MMSE-ESR_{time_id}.png',dpi=100, bbox_inches='tight') 
    plt.show()

    return ser





def complex_to_vec(X):
    "Converts complex matrix to real vector"
    X_vec = np.concatenate([np.real(X) , np.imag(X) ], axis=2)
    return X_vec


def lmmse_channel_estimation(x, y, snr):
    """LMMSE channel estimation"""
    sigma2 = 10 ** (-snr / 10)
    K = x.shape[0]
    cov = np.conj(x).T @ x + sigma2
    inv_term = np.linalg.pinv(cov)
    h = inv_term @ np.conj(x).T @ y
    return h.T





def predict_symbol(h_est, y, constellation):
    """Predict symbol for MIMO system"""
    num_ant = h_est.shape[0]
    num_const = len(constellation)
    all_combinations = np.array(np.meshgrid(*[constellation] * num_ant)).T.reshape(-1, num_ant)
    y_reshaped = y.reshape(-1, 1)
    predicted_y = np.dot(h_est, all_combinations.T)
    distances = np.sum(np.abs(y_reshaped - predicted_y) ** 2, axis=0)
    best_combination_idx = np.argmin(distances)
    return all_combinations[best_combination_idx]


def predict_lmmse_know_h(H, x, y, snr,constellation):
    num_ant = H.shape[0]
    num_const = len(constellation)
    all_combinations = np.array(np.meshgrid(*[constellation] * num_ant)).T.reshape(-1, num_ant)
    sigma2 = 10 ** (-snr / 10)
    I = np.eye(H.shape[0])
    H_H = np.conj(H.T)
    inverse_term = np.linalg.inv(2 * sigma2 * I + H_H @ H)
    H_H_y = H_H @ y
    x_hat_tau_lin = inverse_term @ H_H_y

    distances = np.sum(np.abs(x_hat_tau_lin - all_combinations) ** 2, axis=1)

    best_combination_idx = np.argmin(distances)
    return all_combinations[best_combination_idx]



# all_combinations = np.array(np.meshgrid(*[args.constellation] * args.num_ant)).T.reshape(-1, args.num_ant)
#MMSE estimate of x
def mmse_estimation_and_ser(args, batch_size, k, snr):
    sigma2 = 10 ** (-snr / 10)
    args.SNR_dB_min = snr
    args.SNR_dB_max = snr
    ntask = batch_size*10
    X, Y,_, Hs = generate_signals(batch_size, ntask, args, seed_task=0, seed_example=0)
    assert X.shape == (batch_size, args.prompt_seq_length, args.num_ant)
    assert Y.shape == (batch_size, args.prompt_seq_length, args.num_ant)

    X = X[:, :k + 1, :]
    Y = Y[:, :k + 1, :]
    ser_sum = 0
    mse_sum = 0
    _, constellation = generate_modulated_signal(args, args.modulation)
    all_combinations = np.array(list(product(constellation, repeat=args.num_ant)))
    for i in range(batch_size):
        x_pilot = X[i, :k, :] 
        y_pilot = Y[i, :k, :]
        y_target = Y[i, k, :]  

        likelihoods = []
        for x in all_combinations:
            X_full = np.vstack((x_pilot, x[np.newaxis, :]))
            y_full = Y[i, :k + 1, :]

            C = np.zeros((args.num_ant * (k + 1), args.num_ant * (k + 1)), dtype=complex)
            C = np.outer(X_full, X_full.conj())+ sigma2 * np.eye(args.num_ant * (k + 1))


            likelihood = np.exp(-0.5 * y_full.conj().T @ linalg.inv(C) @ y_full)
            likelihood /= np.sqrt(np.abs(linalg.det(C)))
            likelihoods.append(likelihood)


        likelihoods = np.array(likelihoods).reshape(-1)
        likelihoods = np.sum(likelihoods)

        x_est = np.sum(all_combinations * likelihoods[:, np.newaxis], axis=0)

        x_decided = all_combinations[np.argmin(np.sum(np.abs(x_est - all_combinations) ** 2, axis=1))]

        ser = np.mean(x_decided != X[i, k, :])
        ser_sum += ser

        mse = np.sum(np.abs(x_est - X[i, k, :]) ** 2)
        mse_sum += mse

    average_mse = mse_sum / batch_size

    print(f"MIMO_{args.num_ant} {args.modulation} Pilot_{k}  SNR_{snr} dB x_MMSE MSE_"f"{average_mse:.4f}")

    return average_mse

