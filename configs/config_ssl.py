from datetime import datetime

from configs.config_setting import setting_config


class setting_config_ssl(setting_config):
    """Scan-aware semi-supervised training on top of the original VM-UNet recipe."""

    labeled_ratio = 0.1
    labeled_batch_size = 8
    unlabeled_batch_size = 24
    ema_decay = 0.99
    cons_weight = 1.0
    dir_weight = 0.5
    rampup_epochs = 50
    conf_hi = 0.8
    conf_lo = 0.2
    ssl_seed = 42
    split_dir = './splits'
    use_dir_consistency = True
    use_scan_dropout = True
    scan_keep = 2

    work_dir = (
        'results/'
        + setting_config.network
        + '_ssl_'
        + setting_config.datasets
        + f'_r{str(labeled_ratio).replace(".", "p")}_'
        + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
        + '/'
    )
