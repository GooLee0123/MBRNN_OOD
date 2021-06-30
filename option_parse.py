import argparse
import configparser
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


    parser.add_argument('--train',
                        default=False, type=str2bool,
                        dest='train', metavar="True")

    parser.add_argument('--gpuid',
                        default='0', type=str,
                        dest='gpuid', metavar="0")
    parser.add_argument('--gamma',
                        default='0', type=str,
                        dest='gamma', metavar="0")
    parser.add_argument('--log-level',
                        default='info', type=str,
                        dest='log_level', metavar="INFO")
    parser.add_argument('--id',
                        default='galaxy', type=str,
                        dest='id', metavar="galaxy")
    parser.add_argument('--load_key',
                        default='TS2', type=str,
                        dest='load_key', metavar="TS2")

    opt = config_processing(parser.parse_args(remaining_argv))

    outfd_option = ['', 'NC'+str(opt.ncls)]
    if opt.finetune:
        outfd_option.append('ThreeFoldTrain')
    outfd_option.append('Gamma%s' % (str(opt.gamma).replace('.', '_')))

    opt.outfd_prefix = '_'.join(outfd_option)

    ckpt_fd = 'checkpoint'+opt.outfd_prefix
    quant_fd = 'quantity'+opt.outfd_prefix+'_RA_combined_usample_'+opt.load_key

    opt.ckpt_fd = os.path.join(opt.ckpt_dn, ckpt_fd)
    opt.quant_fd = os.path.join(opt.output_dn, opt.quant_dn, quant_fd)

    dirs = [opt.ckpt_fd] if opt.train else [opt.quant_fd]

    make_dirs(dirs)
    return opt


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
