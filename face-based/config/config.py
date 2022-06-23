import os

from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "./config/settings.yaml",
        "./config/.secrets.yaml",
        "./config/settings_sys.yaml",
        "./config/mcfgs.yaml",
    ],
    environments=False,
    load_dotenv=True,
    dotenv_path="config\.env",
)

mdl_configs = Dynaconf(settings_files="./config/mcfgs.yaml")
# print(mdl_configs)
try:
    cfg_mdl = mdl_configs.MODEL_CONFIGS[settings.RUN_CONFIG.model.model_version]
except KeyError:
    raise LookupError(
        f"KeyError: Failed to find config for model_version {settings.model.model_version}"
    ) from None

cfg_sys = settings.SYS_CONFIG
cfg_run = settings.RUN_CONFIG
