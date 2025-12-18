import click
import simplejson as json

from clx.settings import CONFIG_PATH


@click.command()
@click.option(
    "--autoload-env",
    type=bool,
    default=None,
    help="Whether to autoload the .env file with python-dotenv.",
)
def config(autoload_env):
    config = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())

    updated = False

    config["autoload-env"] = config.get("autoload-env", False)
    if autoload_env is not None:
        config["autoload-env"] = autoload_env
        updated = True

    if updated:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(config, indent=4))
        print(f"Config saved to {CONFIG_PATH}")

    print(f"Config: \n{json.dumps(config, indent=4)}")
    print("Run `clx config --help` for more package configuration options.")
