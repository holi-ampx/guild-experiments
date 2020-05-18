import os

import click
import guild.ipy as guild

from main import base_train


@click.command()
@click.option('--toprint', default=10, help='Number of greetings.')
def guild_ipy(*args, **kwargs):
    import shutil
    shutil.rmtree("guild-env")
    if not os.path.exists("guild-env"):
        os.mkdir("guild-env")

    guild.set_guild_home("guild-env")
    #guild._run(base_train, kwargs, {}) # works, but no output captured?
    guild.run(base_train, **kwargs)  # makes it into a batch runner, we dont want it

    """
    kw = kwargs
    opts = guild._pop_opts(kw)
    op = train_lstm_base
    flags = guild._init_flags(op, args, kw)

    def _my_init_runner(op, flags, opts):
        return guild.util.find_apply([guild._single_runner], op, flags, opts)

    run = _my_init_runner(op, flags, opts)
    return run()
    """


@click.group()
def cmd_training():
    pass


cmd_training.add_command(guild_ipy)

if __name__ == "__main__":
    cmd_training()
