import argparse


def get_args():
    """ Gets command-line arguments.

    Returns:
        Return command-line arguments as a set of attributes.
    """

    parser = argparse.ArgumentParser(description='Train WB Correction.')
    parser.add_argument('--do-train', action='store_true', help='Do training')
    parser.add_argument('-trd', '--training-dir', default='./data/images/', type=str,
                        help='Training directory')

    parser.add_argument('--do-eval', action='store_true', help='Do validation')
    parser.add_argument('-valdir', '--validation-dir', dest='valdir',
                        default=None, help='Main validation directory')
    
    parser.add_argument('--do-test', action='store_true', help='Do evaluate model')
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')

    parser.add_argument('-s', '--patch-size', dest='patch_size', type=int,
                        default=64, help='Size of input training patches')

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=8, help='Batch size', dest='batch_size')
    
    parser.add_argument('--img_size', type=bool, default=320, help='keep aspect ratio')

    parser.add_argument('-pn', '--patch-number', type=int, default=4,
                        help='number of patches per trainig image',
                        dest='patch_number')

    parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str,
                        default='Adam', help='Adam or SGD')

    parser.add_argument('-mtf', '--max-tr-files', dest='max_tr_files', type=int,
                        default=0, help='max number of training files; default '
                                        'is 0 which uses all files')

    parser.add_argument('-mvf', '--max-val-files', dest='max_val_files', type=int,
                        default=0, help='max number of validation files; '
                                        'default is 0 which uses all files')

    parser.add_argument('-nrm', '--normalization', dest='norm',
                        action='store_true', help='Apply BN in network')

    parser.add_argument('-msc', '--multi-scale', action='store_true',
                        help='Multi-scale training samples')

    parser.add_argument('-lr', '--lr', metavar='LR', type=float,
                        nargs='?', default=1e-4, help='Learning rate', dest='lr')

    parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                        nargs='?', default=0, help='L2 regularization factor',
                        dest='l2r')

    parser.add_argument('-sw', '--smoothness-weight', dest='smoothness_weight',
                        type=float, default=100.0, help='smoothness weight')

    parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+',
                        # default=['D', 'S', 'T', 'F', 'C'])
                        default=['D', 'S', 'T'])

    parser.add_argument('-l', '--load', action='store_true',
                        help='Load model from a .pth file')

    parser.add_argument('-ml', '--model-location', dest='model_location',
                        default=None, help='Location of the pretrained model')

    parser.add_argument('-vf', '--validation-frequency', dest='val_freq',
                        type=int, default=1, help='Validation frequency.')

    parser.add_argument('--device', nargs='+', default=0, type=int)
    
    parser.add_argument('--accelerator', default='auto', help='cpu, gpu, tpu or auto')

    parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                        default='WB_model', help='Model name')
    
    parser.add_argument('-num-workers', type=int,
                        default=8, help='Number of workers processing')
    
    parser.add_argument('--output_path', default='./output', help='saved logging and dir path')
    parser.add_argument('--aug', action='store_true', help='augmented image')
    parser.add_argument('--multiscale', action='store_true', help='multiscale')
    parser.add_argument('--keep_aspect_ratio', action='store_true', help='keep aspect ratio')


    return parser.parse_args()