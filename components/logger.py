import logging
import logging.config


def init_logger(log_name: str = "eye_measurement.log"):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                'format': '%(name)-12s %(message)s'
            },
            'file': {
                'format': '%(asctime)s %(levelname)-8s %(name)-50s %(lineno)-5d %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'console'
            },
            'file': {
                'level': 'NOTSET',
                'class': 'logging.FileHandler',
                'formatter': 'file',
                'filename': f'{log_name}'
            }
        },
        'loggers': {
            '': {
                'level': 'NOTSET',
                'handlers': ['console', 'file']
            }
        }
    })
