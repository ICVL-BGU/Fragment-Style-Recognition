from frag import init_data_module
from models import *
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml


def get_trainer(config):
    return pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints', filename=config['model_name'], monitor='val_loss', mode='min')]
    )


def get_ckpt_path(config):
    return f'{config["ckpt_dir"]}/{config["model_name"]}.ckpt'


# Trains a new StyleExtrapolator model
def train_style_extrapolator(data_module, config):
    model = StyleExtrapolator(config['lr'])
    trainer = get_trainer(config)
    trainer.fit(model, data_module)
    return model


def train_model(data_module, config):
    # Create the model (`Proposed` models require a pre-trained StyleExtrapolator)
    match config['model_type']:
        case 'proposed':
            if config['sx_path'] == 'new':
                sx = train_style_extrapolator(data_module, config)
            else:
                sx = StyleExtrapolator.load_from_checkpoint(f'{config["ckpt_dir"]}/{config["sx_name"]}.ckpt')
            model = Proposed(sx, config['lr'])
        case 'ft':
            backbone = EfficientNetBackbone(config['n_styles'], config['freeze_upto'])
            model = Baseline(backbone, config['lr'])
        case 'cnn':
            cnn = CNN(config['n_styles'])
            model = Baseline(cnn, config['lr'])
        case _:
            raise ValueError(f'Invalid model type: {config["model_type"]}')

    # Train the model
    trainer = get_trainer(config)
    trainer.fit(model, data_module)
    print(f'Trained checkpoint saved at {get_ckpt_path(config)}')
    return model


def load_model(config):
    if config['model_type'] not in ['proposed', 'ft', 'cnn']:
        raise ValueError(f'Invalid model type: {config["model_type"]}')
    if config['model_type'] == 'proposed':
        model = Proposed.load_from_checkpoint(get_ckpt_path(config))
    else:
        model = Baseline.load_from_checkpoint(get_ckpt_path(config))
    return model


def eval_model(model, data_module):
    trainer = get_trainer(config)
    trainer.test(model, data_module.test_dataloader())



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, default='config.yml', help='path to the configuration file')
    args = parser.parse_args()
    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # Initialize the data module
    data_module = init_data_module(config['data_dir'], config['batch_size'], config['num_workers'])
    # Load model from checkpoint or train a new model
    if not config['do_train']:
        model = load_model(config)
    else:
        model = train_model(data_module, config)
    # Evaluate the model
    if config['do_eval']:
        eval_model(model, data_module)
