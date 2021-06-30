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

    parser.add_argument('--gpuid',
                        default=0, type=str,
                        dest='gpuid', metavar="0")

    parser.add_argument('--train',
                        default=False, type=str2bool,
                        dest='train', metavar="True")
    parser.add_argument('--ensemble',
                        default=False, type=str2bool,
                        dest='ensemble', metavar="False")

    parser.add_argument('--logging',
                        default='file', type=str,
                        dest='logging', metavar="file")
    parser.add_argument('--log-level',
                        default='info', type=str,
                        dest='log_level', metavar="INFO")
    parser.add_argument('--ind',
                        default='galaxy', type=str,
                        dest='ind', metavar="galaxy")
    parser.add_argument('--ul_prefix',
                        default='RA_combined_usample', type=str,
                        dest='ul_prefix', metavar="RA_combined_usample")
    parser.add_argument('--tr_ul_prefix',
                        default='RA_combined_usample', type=str,
                        dest='tr_ul_prefix', metavar="RA_combined_usample")
    parser.add_argument('--load_key',
                        default='fine_tuned', type=str,
                        dest='load_key', metavar="fine_tuned")
    parser.add_argument('--densemble',
                        default='Ensemble', type=str,
                        dest='densemble', metavar="Ensemble")

    parser.add_argument('--gamma',
                        default='0.0', type=str,
                        dest='gamma', metavar="0.0")
    parser.add_argument('--dcp_weight',
                        default='1.0', type=str,
                        dest='dcp_weight', metavar="1.0")

    opt = config_processing(parser.parse_args(remaining_argv))

    ind_prefix = '_ind_'+opt.ind
    opt.data_dn += ind_prefix
    opt.analysis_dn += ind_prefix+'_'+opt.method+'_'+opt.tr_ul_prefix

    outfd_option = ['', 'NC'+str(opt.ncls)]
    if opt.finetune:
        outfd_option.append('ThreeFoldTrain')
    outfd_option.append('Gamma%s' % (str(opt.gamma).replace('.', '_')))
    outfd_option.append('DCPW%s' % (str(opt.dcp_weight).replace('.', '_')))

    opt.outfd_prefix = '_'.join(outfd_option)

    ckpt_fd = 'checkpoint'+opt.outfd_prefix
    loss_fd = 'loss'+opt.outfd_prefix
    quant_fd = 'quantity'+opt.outfd_prefix+'_'+opt.ul_prefix+'_'+opt.load_key
    plot_fd = 'plot'+opt.outfd_prefix+'_'+opt.ul_prefix+'_'+opt.load_key

    opt.ckpt_fd = os.path.join(opt.analysis_dn, opt.ckpt_dn, ckpt_fd)
    opt.loss_fd = os.path.join(opt.analysis_dn, opt.loss_dn, loss_fd)
    opt.quant_fd = os.path.join(opt.analysis_dn, opt.quant_dn, quant_fd)
    opt.plot_fd = os.path.join(opt.analysis_dn, opt.plot_dn, plot_fd)

    dirs = [opt.ckpt_fd] if opt.train \
        else [opt.quant_fd, opt.plot_fd]

    if opt.ensemble:
        eoutfd_option = ['', 'NC'+str(opt.ncls)]
        opt.eoutfd_prefix = '_'.join(eoutfd_option)

        ensemblefd = 'ensemble'+opt.eoutfd_prefix
        opt.ensemblefd = os.path.join(opt.analysis_dn,
                                      opt.densemble, ensemblefd)
        dirs += [opt.ensemblefd]

    make_dirs(dirs)
    return opt


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
