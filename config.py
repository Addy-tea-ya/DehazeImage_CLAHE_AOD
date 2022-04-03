from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='C:/Users/Aditya/FFA-Net/Data/Train/clear',  help='Origin image path')
parser.add_argument('--haze_data_path', type=str, default='C:/Users/Aditya/FFA-Net/Data/Train/hazy',  help='Haze image path')
parser.add_argument('--val_ori_data_path', type=str, default='C:/Users/Aditya/FFA-Net/Data/Test/clear',  help='Validation origin image path')
parser.add_argument('--val_haze_data_path', type=str, default='C:/Users/Aditya/FFA-Net/Data/Test/hazy',  help='Validation haze image path')
parser.add_argument('--sample_output_folder', type=str, default='D:/dataset/output',  help='Validation haze image path')
parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss ')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs for training')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--ckpt', type=str, default='./model/nets1/net_1.pkl')
parser.add_argument('--net_name', type=str, default='nets')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
