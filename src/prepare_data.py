
import argparse
from data.dataloader import DataPreparer, load_config


parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='data',
                    help='Path to output files')
parser.add_argument('--use_packing', type=bool, default=False,
                    help='Set different lookback periods per river catchment')
parser.add_argument('--config', type=str, default='config/4data.yaml',
                    help='Path to YAML config file')

args = parser.parse_args()

cfg = load_config(args.config)
data = DataPreparer(cfg, args)
datatosave = data.prepare_data()
data.save_prepared_data(args.out_dir, **datatosave)

print(f"Prepared data saved to {args.out_dir}")
