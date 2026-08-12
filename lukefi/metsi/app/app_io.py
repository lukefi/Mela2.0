import argparse


# CLI argument parser
def parse_cli_arguments(args: list[str]) -> dict:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Mela2.0 forest growth calculator')
    parser.add_argument('input_path', help='Application input file or directory')
    parser.add_argument('target_directory', help='Directory path for program output')
    parser.add_argument('control_file', nargs='?', help='Application control declaration file')
    parser.add_argument(
        '-d', '--delete',
        action='store_true',
        dest='delete',
        help='If output files already exist, delete them and continue without prompting.'
    )

    return parser.parse_args(args).__dict__
