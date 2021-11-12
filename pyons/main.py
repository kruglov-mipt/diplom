import click
from pyons.models.rfid.main import cli as rfid_cli

@click.group()
def cli():
    pass

cli.add_command(rfid_cli, "rfid")


if __name__ == '__main__':
    cli()
