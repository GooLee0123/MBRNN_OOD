import argparse
import configparser
import logging
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        return v


def config_processing(opts):
    optdict = vars(opts)

    for key in optdict.keys():
        try:
            if '.' in optdict[key]:
                setattr(opts, key, float(optdict[key]))
            else:
                setattr(opts, key, int(optdict[key]))
        except Exception:
            setattr(opts, key, str2bool(optdict[key]))

    return opts


def Parser():
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument('-c', '--conf_file',
                             default='./config_file/config.cfg',
                             help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    Noption = 6
    Keys = ["Training", "Input", "Output", "Network", "Verbose", "Logs"]
    OptionDict = [{}]*Noption

    if args.conf_file:
        config = configparser.SafeConfigParser()
        config.read([args.conf_file])
        for i in range(Noption):
            OptionDict[i].update(dict(config.items(Keys[i])))

    parser = argparse.ArgumentParser(parents=[conf_parser])

    for i in range(Noption):
        parser.set_defaults(**OptionDict[i])

    parser.add_argument('--log-level',
                        default='info', type=str,
                        dest='log_level', metavar="INFO")
    parser.add_argument('--training_stage',
                        default='TS2', type=str,
                        dest='training_stage', metavar="TS2")

    parser.add_argument('--infer', action='store_true', dest='infer')
    parser.add_argument('--test', action='store_true', dest='test')

    opt = config_processing(parser.parse_args(remaining_argv))
    if opt.infer or opt.test:
        opt.train = False

    logging.info(opt)

    outfd_option = ['', 'NC'+str(opt.ncls)]
    outfd_option.append('Gamma%s' % (str(opt.gamma).replace('.', '_')))
    opt.outfd_prefix = '_'.join(outfd_option)

    out_fd = 'output'+opt.outfd_prefix+'_'+opt.training_stage
    ckpt_fd = 'checkpoint'+opt.outfd_prefix

    opt.out_fd = os.path.join(opt.out_dn, out_fd)
    opt.ckpt_fd = os.path.join(opt.ckpt_dn, ckpt_fd)

    dirs = [opt.ckpt_fd] if opt.train else [opt.out_fd]

    make_dirs(dirs)
    return opt


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
