from argparse import ArgumentParser
#get autoargument

def get_arguments():
    """Defines command-line arguments, and parses them."""
    parser = ArgumentParser()
    # Execution mode
    parser.add_argument( "--mode",   choices=['train', 'test', 'single', 'video','vidpred','frames2vid'], default='single')
    parser.add_argument( "--resume", action='store_true')
    parser.add_argument( "--reset-optimizer", dest='reset_optimizer',   action='store_true', help="Reset optimizer to train encoder and decoder.")
    parser.add_argument( "--transfer-learn", dest='transfer_learn',   action='store_true', help="train final layer.")
    # Hyperparameters
    parser.add_argument( "--batch-size",      type=int,   default=8,   help="The batch size. Default: 10")
    parser.add_argument( "--epochs",          type=int,   default=500,  help="Number of training epochs. Default: 300")
    parser.add_argument( "--learning-rate",   type=float, default=5e-4, help="The learning rate. Default: 5e-4")
    parser.add_argument( "--lr-decay",        type=float, default=0.1,  help="The learning rate decay factor. Default: 0.5")
    parser.add_argument( "--lr-decay-epochs", type=int,   default=100,  help="The number of epochs before adjusting the learning rate. Default: 100")
    parser.add_argument( "--weight-decay",    type=float, default=2e-4, help="L2 regularization factor. Default: 2e-4")
    # Dataset
    parser.add_argument( "--dataset",        choices=['camvid', 'cityscapes','ritscapes','kitti'],  default='kitti', help="Dataset to use. Default: cityscapes")
    parser.add_argument( "--dataset-dir",    type=str,        default="/home/ken/Documents/Dataset/")
    parser.add_argument( "--height",         type=int,        default=352,                          help="The image height. Default: 360")
    parser.add_argument( "--width",          type=int,        default=1216,                          help="The image width. Default: 600")
    parser.add_argument( "--weighing",       choices=['enet', 'mfb', 'none'], default='enet',       help="The class weighing technique to apply to the dataset. Default: Enet")
    parser.add_argument( "--with-unlabeled", dest='ignore_unlabeled',         action='store_false', help="The unlabeled class is not ignored.")
    # Settings
    parser.add_argument( "--workers",      type=int,            default=6,            help="Number of subprocesses to use for data loading. Default: 10")
    parser.add_argument( "--print-step",   action='store_true',                        help="Print loss every step")    
    parser.add_argument( "--imshow-batch", action='store_true',                        help=("Displays batch images when loading the dataset and making predictions."))
    parser.add_argument( "--no-cuda",      dest='cuda',         default='store_false', help="CPU only.")
    # Storage settings
    parser.add_argument( "--name",     type=str, default='ERFnet',      help="Name given to the model when saving. Default: ENet")
    parser.add_argument( "--save-dir", type=str, default='save/ERFnet/kitti', help="The directory where models are saved. Default: save")
    parser.add_argument( "--rmdistort", action='store_true')


    return parser.parse_args()
