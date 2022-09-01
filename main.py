import click
import config

from style_transfer import multi_scale_style_transfer

"""
This module accepts arguments from the user and runs the entire code.
The user needs to enter two arguments,
1. The Content images root directory path (in which the content images could be found).
2. The Style images root directory path (in which the style images could be found).

If the user didn't specify any arguments, a default paths for the root directories is taken from the config file.

Dynamic objects:

content_dir_path -- A directory path which contains the content images to draw from to our model
style_dir_path -- A directory path which contains the style images to draw from to our model

Functions:

multi_scale_style_transfer() -- The core function of the code. The function receives as an arguments content and style 
                                paths, and creates new style-transferred images out of them.
"""


def default_callback_builder(message, dir_name):
    def inner():
        click.echo(message)

        if dir_name == 'content':
            return config.content_dir_path
        return config.style_dir_path

    return inner


@click.command()
def info():
    click.echo('In order to get information about running the code, please run the following command:')
    click.echo('python main.py code-run-info')


# @info()
@click.group()
def cli():
    pass


@cli.command(help="This command runs the code")
@click.option('--content_dir_path',
              default=default_callback_builder("\nWarning: Taking content dir path from env variables", "content"),
              help="Content directory's path")
@click.option('--style_dir_path',
              default=default_callback_builder("\nWarning: Taking style dir path from env variables", "style"),
              help="Style directory's path")
@click.option('--number_content_style_images',
              default=1,
              help="Number of Content and Style images which will be randomly chosen to create a Target final image")
def style_transfer(content_dir_path, style_dir_path, number_content_style_images):
    click.echo(f'\nContent path taken: {content_dir_path}')
    click.echo(f'Style path taken: {style_dir_path}')
    click.echo(f'Number of Content and Style images which\'ll be randomly chosen is : {number_content_style_images}\n')

    print('Running the Code...')
    multi_scale_style_transfer(content_dir_path, style_dir_path, number_content_style_images)


@cli.command(help="This command prints out code running information")
def code_run_info():
    click.echo('In order to run the code, please choose a content directory path and style directory path in which the '
               'content and style images could be found\n')
    click.echo('An example command for running the code:')
    click.echo('python main.py style-transfer --content_dir_path=.\\content_dir --style_dir_path=.\\style_dir\n')
    click.echo('Another example command is getting help:')
    click.echo('python main.py\nor\npython main.py --help\n')
    click.echo('For getting help on a specific function:')
    click.echo('python main.py style-transfer --help\nor\npython main.py code-run-info --help')


if __name__ == '__main__':
    cli()

    # If wish to skip the CLI
    # style_transfer()
