import argparse

def parameter_reading():
    parser = argparse.ArgumentParser(
        description="Configuration parameters for training the ICL-Equalizer."
    )

    # --------------------------------------------------------------- #
    # Transformer architecture hyperparameters
    # --------------------------------------------------------------- #
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension of the Transformer input.')
    parser.add_argument('--num_head', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num_layer', type=int, default=8,
                        help='Number of Transformer layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate used inside the Transformer.')

    # --------------------------------------------------------------- #
    # Data / modulation configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--prompt_seq_length', type=int, default=31,
                        help='Sequence length of the transmitted symbols.')
    parser.add_argument('--num_ant', type=int, default=2,
                        help='Number of antennas in the MIMO system.')

    parser.add_argument('--modulation', default='4QAM',
                        help="Modulation type. Options: '4QAM', '16QAM', '64QAM', 'BPSK', '2PSK'.")
    parser.add_argument('--modu_num', type=int, default=4,
                        help='Number of constellation points (will be overwritten by joint constellation).')

    # --------------------------------------------------------------- #
    # Training hyperparameters
    # --------------------------------------------------------------- #
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size used for training.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Total number of training epochs.')
    parser.add_argument('--training_steps', type=int, default=3000,
                        help='Training steps (legacy parameter).')

    # --------------------------------------------------------------- #
    # Channel / SNR configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--SNR_dB_min', type=float, default=20,
                        help='Minimum SNR value in dB.')
    parser.add_argument('--SNR_dB_max', type=float, default=20,
                        help='Maximum SNR value in dB.')

    # --------------------------------------------------------------- #
    # Decision-Feedback training configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--train_pilot_len', type=int, default=1,
                        help='Length of pilot symbols provided to DFE training.')
    parser.add_argument('--DFE_TRAIN', type=bool, default=True,
                        help='Enable two-phase training: ICL → DEFINED.')
    parser.add_argument('--DFE_epoch', type=int, default=1000,
                        help='Epoch at which training switches from ICL to DEFINED.')
    parser.add_argument('--loss_weight', type=float, default=0.7,
                        help='Weight for DEFINED loss: loss = w*loss1 + (1-w)*loss2.')

    # --------------------------------------------------------------- #
    # Miscellaneous
    # --------------------------------------------------------------- #
    parser.add_argument('--model_type', default='GPT2',
                        help='Model type (kept for compatibility).')

    args = parser.parse_args()
    return args