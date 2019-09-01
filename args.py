from argparse import ArgumentParser
#get autoargument

def get_arguments():
    parser = ArgumentParser()
    # Execution mode
    parser.add_argument( "--mode",   choices=['train', 'test', 'single', 'video','vidpred','frames2vid'], default='train')
    parser.add_argument( "--model",   choices=['ERFnet', 'Deeplab'], default='ERFnet')
    parser.add_argument( "--resume", action='store_true')
    # Hyperparameters
    parser.add_argument( "--batch-size",      type=int,   default=1,   help="The batch size. Default: 10")
    parser.add_argument( "--val-batch-size",  type=int,   default=1,   help="The batch size. Default: 10")
    parser.add_argument( "--workers",      type=int,            default=0,   help="Number of subprocesses to use for data loading. Default: 10")
    parser.add_argument( "--epochs",          type=int,   default=500,  help="Number of training epochs. Default: 300")
    parser.add_argument( "--learning-rate",   type=float, default=0.0005, help="The learning rate. Default: 5e-4")
    parser.add_argument( "--lr-decay",        type=float, default=0.5,  help="The learning rate decay factor. Default: 0.5")
    parser.add_argument( "--lr-decay-epochs", type=int,   default=40,  help="The number of epochs before adjusting the learning rate. Default: 100")
    parser.add_argument( "--weight-decay",    type=float, default=1e-4, help="L2 regularization factor. Default: 2e-4")
    parser.add_argument( "--optimizer",   choices=['SGD', 'ADAM'], default='ADAM')
    # Dataset
    parser.add_argument( "--dataset",        choices=['camvid', 'cityscapes','ritscapes','kitti','imagenet'],  default='cityscapes', help="Dataset to use. Default: cityscapes")
    parser.add_argument( "--dataset-dir",    type=str,        default="/home/ken/Documents/Dataset/")
    parser.add_argument( "--height",         type=int,        default=512,                          help="The image height. Default: 360")
    parser.add_argument( "--width",          type=int,        default=1024,                          help="The image width. Default: 600")
    parser.add_argument( "--weighing",       choices=['enet', 'mfb', 'none'], default='enet',       help="The class weighing technique to apply to the dataset. Default: Enet")
    parser.add_argument( "--with-unlabeled", dest='ignore_unlabeled',         action='store_false', help="The unlabeled class is not ignored.")
    # Storage settings
    parser.add_argument( "--name",     type=str, default='ERFnet.pth',      help="Name given to the model when saving. Default: ERFNet")
    parser.add_argument( "--pretrain_name",     type=str, default='erfnet_encoder_pretrained.pth',      help="Name given to the model when saving. Default: ERFNet")
    parser.add_argument( "--save-dir", type=str, default='save/ERFnet', help="The directory where models are saved. Default: save")

    return parser.parse_args()
