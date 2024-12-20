import argparse
import torch
import json
import yaml
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from data_loader.forecasting_dataloader import get_dataloader_forecasting
from tsde.main_model import TSDE_Forecasting
from utils.utils import train, evaluate, gsutil_cp, set_seed

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="TSDE-Forecasting")

parser.add_argument("--run", type=str, default=1)
parser.add_argument("--modelfolder", type=str, default="euro_all_countries/n_samples_100_run_{}_linear_False_sample_feat_True")
parser.add_argument("--dataset", type=str, default='euro_all_countries')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--pred_length', type=int, default=3, help='Prediction length')
parser.add_argument("--nsample", type=int, default=100)
# parser.add_argument('--linear', action='store_true', help='Linear mode flag')
parser.add_argument('--sample_feat', action='store_true', help='Sample feature flag')
# parser.add_argument('--n_nowcast_cols', type=int, default=None, help='number of rightmost columns not to mask for nowcasting')


# parser.add_argument("--dataset", type=str, default='Electricity')
# parser.add_argument("--modelfolder", type=str, default="")
# parser.add_argument("--run", type=int, default=1)
parser.add_argument("--mix_masking_strategy", type=str, default='equal_p', help="Mix masking strategy (equal_p or probabilistic_layering)")

args = parser.parse_args()
print(args)

path = "./save/Forecasting/" + args.modelfolder.format(args.run) 
with open(f'{path}/config.json', 'r') as f:
    config = json.load(f)
# path = "src/config/" + args.config
# with open(path, "r") as f:
#     config = yaml.safe_load(f)

# config["model"]["mix_masking_strategy"] = args.mix_masking_strategy

    

foldername = "./save/Forecasting/" + args.modelfolder.format(args.run) + "/predictions/"
os.makedirs(foldername, exist_ok=True)
print(foldername)
model = TSDE_Forecasting(config, args.device, target_dim=config['embedding']['num_feat'], sample_feat=args.sample_feat).to(args.device)
model.load_state_dict(torch.load(f'{path}/model.pth'))

train_loader, valid_loader, test_loader, scaler, mean_scaler, df_indices, raw_df, predict_loader = get_dataloader_forecasting(
    train_length=-1, # not using
    skip_length=-1, # not using
    valid_length=3, # hack, we're dividing this by 100 to get a percentage
    test_length=10, # hack, we're dividing this by 100 to get a percentage
    pred_length=args.pred_length,
    history_length=36,
    batch_size=config["train"]["batch_size"],
    device=args.device,
    dataset_name='euro_all_countries',
    prediction_run_id = args.run,
)
df_indices.to_pickle(foldername + "df_indices.pkl")
raw_df.to_pickle(foldername + "raw_df.pkl")

os.makedirs(foldername, exist_ok=True)

evaluate(model, predict_loader, args.nsample, foldername=foldername, scaler=scaler, mean_scaler=mean_scaler, save_samples=True)

'''
why is a month cut off from the end of the dataset?
the data has been updated, how can you "update" the data before prediction?

'''