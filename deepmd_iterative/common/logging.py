"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2024/05/01
Last modified: 2024/05/01
"""
from typing import Dict

def setup_logging(verbose: int = 0) -> Dict:
    """
    Setup logging configuration.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level, by default 0
    
    Returns
    -------
    Dict
        Logging configuration
    """
    logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(levelname)s - %(message)s',
                },
                'simple_file': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout'  # Use standard output (or sys.stderr)
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'INFO',
                    'formatter': 'simple_file',
                    'filename': 'arcann.log',
                    'mode': 'a',  # Append mode
                },
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }

    if verbose >= 1:
        logging_config['handlers']['console']['level'] = 'DEBUG'
        logging_config['handlers']['console']['formatter'] = 'detailed'
        logging_config['handlers']['file']['level'] = 'DEBUG'
        logging_config['handlers']['file']['formatter'] = 'detailed'
        logging_config['loggers']['']['level'] = 'DEBUG'

    return logging_config